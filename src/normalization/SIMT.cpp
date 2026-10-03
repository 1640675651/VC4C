/*
 * See the file "LICENSE" for the full license governing this code.
 */

#include "SIMT.h"

#include "../InstructionWalker.h"
#include "../Method.h"
#include "../Module.h"
#include "../intermediate/IntermediateInstruction.h"
#include "../intrinsics/WorkItems.h"
#include "log.h"

#include <cstdlib>
#include <string>

using namespace vc4c;
using namespace vc4c::intermediate;

// Number of work-items per QPU in SIMT mode
static constexpr unsigned char SIMT_WIDTH = NATIVE_VECTOR_SIZE;

namespace
{
    /*
     * How a work-item dependent ("varying") value differs between the lanes: if known, the value in lane i is the
     * value in lane 0 plus i * stride.
     */
    struct LaneInfo
    {
        Optional<int64_t> stride;

        bool operator==(const LaneInfo& other) const
        {
            return stride == other.stride;
        }
    };

    struct Analysis
    {
        FastMap<const Local*, LaneInfo> varying;

        bool isVarying(const Value& val) const
        {
            auto loc = val.checkLocal();
            return loc && varying.find(loc) != varying.end();
        }

        Optional<int64_t> getStride(const Value& val) const
        {
            if(auto loc = val.checkLocal())
            {
                auto it = varying.find(loc);
                if(it != varying.end())
                    return it->second.stride;
            }
            // uniform values (and literals) do not differ between lanes
            return int64_t{0};
        }
    };
} // namespace

static bool isSupportedType(const DataType& type)
{
    if(type.getPointerType())
        return true;
    if(type.isLabelType() || type.isVoidType())
        return true;
    return type.isScalarType() && type.getScalarBitCount() <= 32;
}

static bool isWorkItemIdCall(const MethodCall& call)
{
    return call.methodName == intrinsics::FUNCTION_NAME_LOCAL_ID ||
        call.methodName == intrinsics::FUNCTION_NAME_GLOBAL_ID;
}

// The constant dimension argument of a work-item function call, if any
static Optional<int32_t> getDimension(const MethodCall& call)
{
    auto arg = call.getArgument(0);
    // after inlining, the dimension is usually a local set once from the constant, e.g. the parameter of the inlined
    // get_global_id(uint)
    for(unsigned i = 0; arg && i < 4; ++i)
    {
        if(auto lit = arg->getLiteralValue())
            return lit->signedInt();
        auto loc = arg->checkLocal();
        auto writer = loc ? dynamic_cast<const MoveOperation*>(loc->getSingleWriter()) : nullptr;
        if(!writer || writer->hasConditionalExecution())
            return {};
        arg = writer->getSource();
    }
    return {};
}

/*
 * Rejects kernels using functionality which is not (yet) supported in SIMT mode. Returns the reason, or an empty
 * string if the kernel might be supported.
 */
static std::string checkUnsupported(const Method& method)
{
    const auto& sizes = method.metaData.workGroupSizes;
    if(sizes[0] != 0 && (sizes[0] > SIMT_WIDTH || sizes[1] > 1 || sizes[2] > 1))
        return "required work-group size is larger than 16 or not 1-dimensional";
    if(!method.stackAllocations.empty())
        return "uses private memory (stack allocations)";
    for(const auto& param : method.parameters)
    {
        if(!isSupportedType(param.type))
            return "parameter type " + param.type.to_string();
        if(auto ptrType = param.type.getPointerType())
        {
            if(ptrType->addressSpace == AddressSpace::LOCAL)
                return "uses __local memory";
        }
    }

    for(const auto& block : method)
    {
        for(const auto& instr : block)
        {
            if(!instr)
                continue;
            if(dynamic_cast<const MemoryBarrier*>(instr.get()) || dynamic_cast<const SemaphoreAdjustment*>(instr.get()) ||
                dynamic_cast<const MutexLock*>(instr.get()))
                return "uses synchronization: " + instr->to_string();
            if(auto call = dynamic_cast<const MethodCall*>(instr.get()))
            {
                for(const char* unsupported :
                    {"barrier", "atomic", "mutex", "semaphore", "dma", "vpm", "fence", "async", "prefetch", "linear_id"})
                {
                    if(call->methodName.find(unsupported) != std::string::npos)
                        return "calls " + call->methodName;
                }
                if(isWorkItemIdCall(*call) && !getDimension(*call))
                    return "work-item ID with non-constant dimension";
            }
            if(auto mem = dynamic_cast<const MemoryInstruction*>(instr.get()))
            {
                if(mem->op != MemoryOperation::READ && mem->op != MemoryOperation::WRITE)
                    return "memory copy or fill: " + mem->to_string();
                auto numEntries = mem->getNumEntries().getLiteralValue();
                if(!numEntries || numEntries->signedInt() != 1)
                    return "multi-element memory access: " + mem->to_string();
            }
            bool unsupportedType = false;
            instr->forUsedLocals([&](const Local* loc, LocalUse::Type, const IntermediateInstruction&) {
                if(!isSupportedType(loc->type))
                    unsupportedType = true;
            });
            if(unsupportedType)
                return "uses vector or 64-bit values: " + instr->to_string();
        }
    }
    return "";
}

static void markVarying(Analysis& analysis, const Local* loc, LaneInfo info, bool& changed)
{
    auto it = analysis.varying.find(loc);
    if(it == analysis.varying.end())
    {
        analysis.varying.emplace(loc, info);
        changed = true;
    }
    else if(it->second.stride && !(it->second == info))
    {
        // written with different strides (e.g. by several phi-node moves)
        it->second.stride = {};
        changed = true;
    }
}

static Optional<int64_t> scaleStride(Optional<int64_t> stride, int64_t factor)
{
    if(!stride)
        return {};
    return *stride * factor;
}

/*
 * Determines the lane info of the output of an instruction reading a varying value (or depending on varying flags).
 */
static LaneInfo determineLaneInfo(const Analysis& analysis, const IntermediateInstruction& instr, bool varyingFlags)
{
    if(varyingFlags && instr.hasConditionalExecution())
        // e.g. a select: the value per lane depends on the per-lane condition
        return LaneInfo{};

    auto arg0 = instr.getArgument(0);
    auto arg1 = instr.getArgument(1);
    if(auto move = dynamic_cast<const MoveOperation*>(&instr))
    {
        if(!move->getVectorRotation() && !instr.hasUnpackMode() && !instr.hasPackMode())
            return LaneInfo{analysis.getStride(move->getSource())};
        return LaneInfo{};
    }
    if(auto op = dynamic_cast<const Operation*>(&instr))
    {
        if(instr.hasUnpackMode() || instr.hasPackMode() || !arg0)
            return LaneInfo{};
        if(op->op == OP_ADD && arg1)
        {
            auto s0 = analysis.getStride(*arg0);
            auto s1 = analysis.getStride(*arg1);
            if(s0 && s1)
                return LaneInfo{*s0 + *s1};
        }
        if(op->op == OP_SUB && arg1)
        {
            auto s0 = analysis.getStride(*arg0);
            auto s1 = analysis.getStride(*arg1);
            if(s0 && s1)
                return LaneInfo{*s0 - *s1};
        }
        if(op->op == OP_SHL && arg1)
        {
            auto shift = arg1->getLiteralValue();
            if(shift && !analysis.isVarying(*arg1) && shift->signedInt() >= 0 && shift->signedInt() < 31)
                return LaneInfo{scaleStride(analysis.getStride(*arg0), int64_t{1} << shift->signedInt())};
        }
        return LaneInfo{};
    }
    if(auto intrinsic = dynamic_cast<const IntrinsicOperation*>(&instr))
    {
        if(intrinsic->opCode == "mul" && arg0 && arg1)
        {
            if(auto factor = arg1->getLiteralValue())
                return LaneInfo{scaleStride(analysis.getStride(*arg0), factor->signedInt())};
            if(auto factor = arg0->getLiteralValue())
                return LaneInfo{scaleStride(analysis.getStride(*arg1), factor->signedInt())};
        }
        if((intrinsic->opCode == "zext" || intrinsic->opCode == "sext") && arg0)
        {
            // An extension only keeps the stride if the narrow value can't wrap around between the lanes. This is
            // guaranteed for the work-item IDs (e.g. the 8-bit local ID), which are below 16 in SIMT mode.
            auto loc = arg0->checkLocal();
            auto call = loc ? dynamic_cast<const MethodCall*>(loc->getSingleWriter()) : nullptr;
            if(call && isWorkItemIdCall(*call))
                return LaneInfo{analysis.getStride(*arg0)};
        }
        return LaneInfo{};
    }
    return LaneInfo{};
}

/*
 * Determines all values which differ between the work-items of a QPU.
 */
static Analysis analyzeVaryingValues(Method& method)
{
    Analysis analysis;
    bool changed = true;
    while(changed)
    {
        changed = false;
        for(auto& block : method)
        {
            // whether the flags were last set from a varying value
            bool varyingFlags = false;
            for(auto& instr : block)
            {
                if(!instr)
                    continue;
                auto out = instr->checkOutputLocal();
                if(auto call = dynamic_cast<const MethodCall*>(instr.get()))
                {
                    if(isWorkItemIdCall(*call))
                    {
                        auto dim = getDimension(*call);
                        if(dim && *dim == 0 && out)
                            markVarying(analysis, out, LaneInfo{int64_t{1}}, changed);
                        // dimensions 1 and 2 are uniform, since SIMT mode is only used for 1-dimensional work-groups
                        continue;
                    }
                }

                bool readsVarying = false;
                instr->forReadLocals([&](const Local* loc, const IntermediateInstruction&) {
                    if(analysis.varying.find(loc) != analysis.varying.end())
                        readsVarying = true;
                });
                auto mem = dynamic_cast<const MemoryInstruction*>(instr.get());
                if(mem && mem->op == MemoryOperation::READ)
                {
                    // the loaded value is varying if the address is
                    if(analysis.isVarying(mem->getSource()))
                        markVarying(analysis, mem->getDestination().checkLocal(), LaneInfo{}, changed);
                }
                else if(mem)
                {
                    // the "output" of a memory write is the address, which is not modified
                }
                else if(out && (readsVarying || (varyingFlags && instr->hasConditionalExecution())))
                    markVarying(analysis, out, determineLaneInfo(analysis, *instr, varyingFlags), changed);

                if(instr->doesSetFlag())
                    varyingFlags = readsVarying;
            }
        }
    }
    return analysis;
}

/*
 * Checks whether the kernel can be run in SIMT mode with the given varying values. Returns the reason why not, or an
 * empty string.
 */
static std::string checkVaryingValues(const Method& method, const Analysis& analysis)
{
    for(const auto& block : method)
    {
        bool varyingFlags = false;
        for(const auto& instr : block)
        {
            if(!instr)
                continue;
            bool readsVarying = false;
            instr->forReadLocals([&](const Local* loc, const IntermediateInstruction&) {
                if(analysis.varying.find(loc) != analysis.varying.end())
                    readsVarying = true;
            });

            if(auto branch = dynamic_cast<const Branch*>(instr.get()))
            {
                if(!branch->isUnconditional() && varyingFlags)
                    return "divergent branch: " + branch->to_string();
            }
            if(auto mem = dynamic_cast<const MemoryInstruction*>(instr.get()))
            {
                const auto& address = mem->op == MemoryOperation::READ ? mem->getSource() : mem->getDestination();
                auto elementType =
                    mem->op == MemoryOperation::READ ? mem->getSourceElementType() : mem->getDestinationElementType();
                if(analysis.isVarying(address))
                {
                    auto stride = analysis.getStride(address);
                    if(!stride || !elementType.isScalarType() ||
                        *stride != static_cast<int64_t>(elementType.getScalarBitCount() / 8))
                        return "memory access which is not contiguous across work-items: " + mem->to_string();
                }
                else if(mem->op == MemoryOperation::WRITE && analysis.isVarying(mem->getSource()))
                    return "work-items writing different values to the same address: " + mem->to_string();
            }
            if(readsVarying && !dynamic_cast<const Operation*>(instr.get()) &&
                !dynamic_cast<const MoveOperation*>(instr.get()) &&
                !dynamic_cast<const IntrinsicOperation*>(instr.get()) &&
                !dynamic_cast<const MemoryInstruction*>(instr.get()) && !dynamic_cast<const MethodCall*>(instr.get()))
                return "unsupported instruction reading a work-item dependent value: " + instr->to_string();
            if(auto move = dynamic_cast<const MoveOperation*>(instr.get()))
            {
                if(readsVarying && move->getVectorRotation())
                    return "vector rotation of a work-item dependent value: " + move->to_string();
            }

            if(instr->doesSetFlag())
                varyingFlags = readsVarying;
        }
    }
    return "";
}

/*
 * Converts the method: varying values become vectors with one element per work-item.
 */
static void convertToSIMT(Method& method, const Analysis& analysis)
{
    // 1. new vector locals for all varying non-pointer values. Pointers keep their type: they hold the per-lane
    // address, but only the address in lane 0 is used by the (converted) memory accesses below.
    FastMap<const Local*, Value> replacements;
    for(const auto& entry : analysis.varying)
    {
        const Local* loc = entry.first;
        if(loc->type.getPointerType())
            continue;
        replacements.emplace(loc, method.addNewLocal(loc->type.toVectorType(SIMT_WIDTH), loc->name, "simt"));
    }

    for(auto& block : method)
    {
        auto it = block.walk().nextInBlock();
        while(!it.isEndOfBlock())
        {
            auto instr = it.get();
            if(!instr)
            {
                it.nextInBlock();
                continue;
            }

            // 2. work-item IDs in dimension 0: the ID of the QPU's first work-item (as calculated today, since the
            // run-time passes the first local ID of every QPU) plus the element number
            auto call = it.get<MethodCall>();
            if(call && isWorkItemIdCall(*call))
            {
                auto out = call->checkOutputLocal();
                auto replacement = out ? replacements.find(out) : replacements.end();
                if(replacement != replacements.end())
                {
                    auto base = method.addNewLocal(out->type, out->name, "simt_base");
                    call->setOutput(base);
                    auto next = it.copy().nextInBlock();
                    next.emplace(std::make_unique<Operation>(
                        OP_ADD, Value(replacement->second), Value(base), Value(ELEMENT_NUMBER_REGISTER)));
                    CPPLOG_LAZY(logging::Level::DEBUG,
                        log << "SIMT: work-item ID per lane: " << next->to_string() << logging::endl);
                    it.nextInBlock().nextInBlock();
                    continue;
                }
            }

            // 3. memory accesses with per-lane contiguous addresses: access a vector of 16 elements at the address of
            // lane 0
            if(auto mem = it.get<MemoryInstruction>())
            {
                bool isRead = mem->op == MemoryOperation::READ;
                const Value address = isRead ? mem->getSource() : mem->getDestination();
                if(analysis.isVarying(address))
                {
                    auto ptrType = address.type.getPointerType();
                    auto elementType = isRead ? mem->getSourceElementType() : mem->getDestinationElementType();
                    auto vectorPtr = method.addNewLocal(
                        method.createPointerType(
                            elementType.toVectorType(SIMT_WIDTH), ptrType->addressSpace, ptrType->alignment),
                        address.checkLocal()->name, "simt_ptr");
                    it.emplace(std::make_unique<MoveOperation>(vectorPtr, address));
                    it.nextInBlock();
                    mem = it.get<MemoryInstruction>();
                    // the address is the source of a read and the output of a write
                    if(isRead)
                        mem->setArgument(0, vectorPtr);
                    else
                        mem->setOutput(vectorPtr);
                    CPPLOG_LAZY(logging::Level::DEBUG,
                        log << "SIMT: memory access per lane: " << mem->to_string() << logging::endl);
                }
            }
            it.nextInBlock();
        }
    }

    // 4. redirect all uses of the varying values to their vector versions
    for(auto& block : method)
    {
        for(auto& instr : block)
        {
            if(!instr)
                continue;
            for(const auto& replacement : replacements)
            {
                if(instr->readsLocal(replacement.first) || instr->writesLocal(replacement.first))
                    instr->replaceLocal(replacement.first, replacement.second.checkLocal());
            }
        }
    }

    // 5. memory stores only write back the elements of work-items in the work-group, which needs the local size (see
    // lowerWriteRAM in periphery/VPM.cpp). Read it here, so the UNIFORM is loaded at the start of the kernel.
    auto startIt = method.walkAllInstructions();
    if(!startIt.isEndOfMethod() && startIt.get<BranchLabel>())
        startIt.nextInBlock();
    startIt.emplace(std::make_unique<MoveOperation>(method.addNewLocal(TYPE_INT32, "%simt_local_sizes"),
        method.findOrCreateBuiltin(BuiltinLocal::Type::LOCAL_SIZES)->createReference()));

    method.metaData.mergedWorkItemsFactor = SIMT_WIDTH;
}

static bool isSIMTEnabled(const Configuration& config)
{
    const std::string name = normalization::SIMT_PASS_NAME;
    if(config.additionalDisabledOptimizations.find(name) != config.additionalDisabledOptimizations.end())
        return false;
    if(config.additionalEnabledOptimizations.find(name) != config.additionalEnabledOptimizations.end())
        return true;
    return std::getenv("VC4C_NO_SIMT") == nullptr;
}

void normalization::vectorizeWorkItems(Module& module, Method& method, const Configuration& config)
{
    if(!isSIMTEnabled(config))
    {
        CPPLOG_LAZY(logging::Level::DEBUG,
            log << "SIMT: disabled, not checking kernel '" << method.name << "'" << logging::endl);
        return;
    }

    auto reason = checkUnsupported(method);
    if(reason.empty())
    {
        auto analysis = analyzeVaryingValues(method);
        CPPLOG_LAZY_BLOCK(logging::Level::DEBUG, {
            for(const auto& entry : analysis.varying)
            {
                auto writer = entry.first->getSingleWriter();
                logging::debug() << "SIMT: work-item dependent " << entry.first->to_string() << ", lane stride "
                                 << (entry.second.stride ? std::to_string(*entry.second.stride) : "unknown")
                                 << ", written by: " << (writer ? writer->to_string() : "(several)") << logging::endl;
            }
        });
        if(analysis.varying.empty())
            reason = "does not use work-item IDs";
        else
            reason = checkVaryingValues(method, analysis);
        if(reason.empty())
        {
            convertToSIMT(method, analysis);
            CPPLOG_LAZY(logging::Level::INFO,
                log << "SIMT: running kernel '" << method.name << "' with one work-item per SIMD lane ("
                    << analysis.varying.size() << " work-item dependent values)" << logging::endl);
            return;
        }
    }
    CPPLOG_LAZY(logging::Level::INFO,
        log << "SIMT: not using SIMT mode for kernel '" << method.name << "': " << reason << logging::endl);
}
