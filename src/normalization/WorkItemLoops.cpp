/*
 * Author: doe300
 *
 * See the file "LICENSE" for the full license governing this code.
 */

#include "WorkItemLoops.h"

#include "../GlobalValues.h"
#include "../InstructionWalker.h"
#include "../Method.h"
#include "../intermediate/Helper.h"
#include "../intermediate/VectorHelper.h"
#include "../intermediate/operators.h"
#include "../intrinsics/WorkItems.h"
#include "config.h"
#include "log.h"

#include <algorithm>
#include <functional>
#include <cstdlib>
#include <string>
#include <vector>

using namespace vc4c;
using namespace vc4c::intermediate;
using namespace vc4c::normalization;
using namespace vc4c::operators;

const char* const normalization::WORK_ITEM_LOOPS_PASS_NAME = "work-item-loops";

static const std::string BARRIER_FUNCTION = "vc4cl_barrier";
static const std::string WORK_ITEM_LOOP_BARRIER_FUNCTION = "vc4cl_work_item_loop_barrier";

// The maximum number of work-items per QPU: one per lane of the vector registers keeping their values
static constexpr uint32_t MAX_WORK_ITEMS_PER_QPU = NATIVE_VECTOR_SIZE;
// The maximum number of values kept per work-item across a barrier, i.e. of vector registers used for them
static constexpr std::size_t MAX_CONTEXT_VALUES = 16;

static bool isWorkItemLoopsEnabled(const Configuration& config)
{
    const std::string name = WORK_ITEM_LOOPS_PASS_NAME;
    if(config.additionalDisabledOptimizations.find(name) != config.additionalDisabledOptimizations.end())
        return false;
    if(config.additionalEnabledOptimizations.find(name) != config.additionalEnabledOptimizations.end())
        return true;
    return std::getenv("VC4C_NO_WORK_ITEM_LOOPS") == nullptr;
}

static bool isBarrier(const IntermediateInstruction* instr)
{
    auto call = dynamic_cast<const MethodCall*>(instr);
    return call && call->methodName == BARRIER_FUNCTION;
}

// Whether the local holds a value of the kernel code (not a parameter, global, label, etc.)
static bool isCodeLocal(const Local* loc)
{
    return loc && !loc->is<Parameter>() && !loc->is<Global>() && !loc->is<BuiltinLocal>() &&
        !loc->is<StackAllocation>() && !loc->type.isLabelType();
}

static std::vector<BasicBlock*> getSuccessors(BasicBlock& block)
{
    std::vector<BasicBlock*> successors;
    block.forSuccessiveBlocks([&](BasicBlock& next, InstructionWalker) {
        if(std::find(successors.begin(), successors.end(), &next) == successors.end())
            successors.push_back(&next);
    });
    return successors;
}

static bool endsKernel(const BasicBlock& block)
{
    for(const auto& instr : block)
    {
        if(dynamic_cast<const Return*>(instr.get()))
            return true;
    }
    return false;
}

/*
 * Calls the consumers for the values of the kernel code the instruction reads and for the one it (unconditionally)
 * writes. Memory accesses other than loads have their address as output, which they read.
 */
static void forReadAndWrittenLocals(const IntermediateInstruction& instr,
    const std::function<void(const Local*)>& onRead, const std::function<void(const Local*)>& onWrite)
{
    auto mem = dynamic_cast<const MemoryInstruction*>(&instr);
    if(mem && mem->op != MemoryOperation::READ)
    {
        instr.forUsedLocals([&](const Local* loc, LocalUse::Type, const IntermediateInstruction&) {
            if(isCodeLocal(loc))
                onRead(loc);
        });
        return;
    }
    instr.forUsedLocals([&](const Local* loc, LocalUse::Type type, const IntermediateInstruction&) {
        if(isCodeLocal(loc) && has_flag(type, LocalUse::Type::READER))
            onRead(loc);
    });
    if(auto out = instr.checkOutputLocal())
    {
        // The (conditional) writes of phi values in the predecessor blocks write the value for the successor taking it,
        // on the other path it is not read before being written again
        if(isCodeLocal(out) &&
            (!instr.hasConditionalExecution() || instr.hasDecoration(InstructionDecorations::PHI_NODE)))
            onWrite(out);
    }
}

/*
 * Block-level liveness of the values of the kernel code: the locals read before being (unconditionally) written, for
 * the start of every block.
 */
static FastMap<const BasicBlock*, FastSet<const Local*>> determineLiveIns(Method& method)
{
    FastMap<const BasicBlock*, FastSet<const Local*>> uses;
    FastMap<const BasicBlock*, FastSet<const Local*>> definitions;
    FastMap<BasicBlock*, std::vector<BasicBlock*>> successors;
    for(auto& block : method)
    {
        auto& use = uses[&block];
        auto& definition = definitions[&block];
        for(const auto& instr : block)
        {
            if(!instr)
                continue;
            forReadAndWrittenLocals(
                *instr,
                [&](const Local* loc) {
                    if(definition.find(loc) == definition.end())
                        use.emplace(loc);
                },
                [&](const Local* loc) { definition.emplace(loc); });
        }
        successors.emplace(&block, getSuccessors(block));
    }

    FastMap<const BasicBlock*, FastSet<const Local*>> liveIns;
    bool changed = true;
    while(changed)
    {
        changed = false;
        for(auto& block : method)
        {
            FastSet<const Local*> live = uses[&block];
            for(auto next : successors[&block])
            {
                for(auto loc : liveIns[next])
                {
                    if(definitions[&block].find(loc) == definitions[&block].end())
                        live.emplace(loc);
                }
            }
            auto& current = liveIns[&block];
            if(live.size() != current.size())
            {
                current = std::move(live);
                changed = true;
            }
        }
    }
    return liveIns;
}

/*
 * Determines whether a value is the same for all work-items and does not change while the kernel runs: it is written
 * once, computed only from literals, parameters, work-group uniform work-item functions and other such values. Such
 * values need not be kept per work-item. E.g. values depending on a loop counter (written several times) do change:
 * every work-item running a region recomputes them from its own loop counter, overwriting the value of the others.
 */
class InvariantValues
{
public:
    bool isInvariant(const Value& val)
    {
        if(val.getLiteralValue() || val.checkVector())
            return true;
        auto loc = val.checkLocal();
        if(!loc)
            // e.g. registers like the element number
            return false;
        if(loc->is<Parameter>() || loc->is<Global>())
            return true;
        if(!isCodeLocal(loc) || loc->countUsers(LocalUse::Type::WRITER) != 1)
            return false;
        auto it = invariant.find(loc);
        if(it != invariant.end())
            return it->second;
        // assume not invariant while checking (cycles)
        invariant[loc] = false;
        auto writer = loc->getSingleWriter();
        bool result = writer && isInvariantWrite(*writer);
        invariant[loc] = result;
        return result;
    }

private:
    FastMap<const Local*, bool> invariant;

    bool isInvariantWrite(const IntermediateInstruction& writer)
    {
        if(writer.hasConditionalExecution() || writer.hasSideEffects())
            return false;
        if(auto call = dynamic_cast<const MethodCall*>(&writer))
        {
            const std::string& name = call->methodName;
            bool uniformFunction = name == intrinsics::FUNCTION_NAME_LOCAL_SIZE ||
                name == intrinsics::FUNCTION_NAME_NUM_GROUPS || name == intrinsics::FUNCTION_NAME_GROUP_ID ||
                name == intrinsics::FUNCTION_NAME_GLOBAL_OFFSET || name == intrinsics::FUNCTION_NAME_GLOBAL_SIZE ||
                name == intrinsics::FUNCTION_NAME_NUM_DIMENSIONS;
            if(!uniformFunction)
                return false;
        }
        else if(!dynamic_cast<const MoveOperation*>(&writer) && !dynamic_cast<const Operation*>(&writer) &&
            !dynamic_cast<const IntrinsicOperation*>(&writer))
            // e.g. memory loads
            return false;
        for(const auto& arg : writer.getArguments())
        {
            if(!isInvariant(arg))
                return false;
        }
        return true;
    }
};

// Whether the kernel uses __local memory (parameters or variables), which prevents running its work-items independently
static bool usesLocalMemory(const Method& method)
{
    auto isLocalMemory = [](const DataType& type) -> bool {
        auto ptrType = type.getPointerType();
        return ptrType && ptrType->addressSpace == AddressSpace::LOCAL;
    };
    for(const auto& param : method.parameters)
    {
        if(isLocalMemory(param.type))
            return true;
    }
    for(const auto& block : method)
    {
        for(const auto& instr : block)
        {
            if(!instr)
                continue;
            bool found = false;
            instr->forUsedLocals([&](const Local* loc, LocalUse::Type, const IntermediateInstruction&) {
                if(isLocalMemory(loc->type))
                    found = true;
            });
            if(found)
                return true;
        }
    }
    return false;
}

struct BarrierInfo
{
    // the block starting with the barrier call
    BasicBlock* block;
    // the values to keep per work-item across the barrier
    std::vector<const Local*> contextValues;
};

static std::string checkKernel(Method& method, std::vector<BarrierInfo>& barriers)
{
    if(!method.stackAllocations.empty())
        return "uses private arrays (stack allocations)";
    if(method.metaData.mergedWorkItemsFactor > 1)
        return "runs in SIMT mode";

    // split the blocks so every barrier starts its own block
    std::vector<InstructionWalker> barrierCalls;
    for(auto& block : method)
    {
        for(auto it = block.walk(); !it.isEndOfBlock(); it.nextInBlock())
        {
            if(!it.has())
                continue;
            if(isBarrier(it.get()))
                barrierCalls.push_back(it);
            else if(auto call = dynamic_cast<const MethodCall*>(it.get()))
            {
                // work-group functions are lowered with their own barriers between the work-items of a work-group
                for(const char* unsupported : {"async", "wait_group_events", "prefetch", "work_group"})
                {
                    if(call->methodName.find(unsupported) != std::string::npos)
                        return "calls " + call->methodName;
                }
            }
        }
    }
    // Kernels without barriers and __local memory run their work-items independently (in chunks) anyway. Kernels
    // without barriers, but with __local memory, loop over their work-items in a single region.
    if(barrierCalls.empty() && !usesLocalMemory(method))
        return "has no barriers";
    // back to front, since splitting moves the following instructions of the block
    for(auto callIt = barrierCalls.rbegin(); callIt != barrierCalls.rend(); ++callIt)
    {
        if(callIt->copy().previousInBlock().isStartOfBlock())
            continue;
        auto label = method.addNewLocal(TYPE_LABEL, "%work_item_loop_barrier").local();
        auto labelIt = method.emplaceLabel(*callIt, std::make_unique<BranchLabel>(*label));
        static_cast<void>(labelIt);
    }
    // a barrier at the very start of the kernel has nothing to synchronize
    {
        auto it = method.begin()->walk().nextInBlock();
        if(!it.isEndOfBlock() && it.has() && isBarrier(it.get()))
            it.erase();
    }

    bool hasBarrier = false;
    for(auto& block : method)
    {
        auto it = block.walk().nextInBlock();
        if(it.isEndOfBlock() || !it.has() || !isBarrier(it.get()))
            continue;
        hasBarrier = true;
    }
    if(!hasBarrier && !usesLocalMemory(method))
        return "has no barriers";

    auto liveIns = determineLiveIns(method);
    InvariantValues invariantValues;
    // in the order of the blocks
    for(auto& block : method)
    {
        auto it = block.walk().nextInBlock();
        if(it.isEndOfBlock() || !it.has() || !isBarrier(it.get()))
            continue;
        BarrierInfo info{&block, {}};
        for(auto loc : liveIns[&block])
        {
            // A value which is the same for all work-items and does not change need not be kept per work-item
            if(invariantValues.isInvariant(loc->createReference()))
                continue;
            // e.g. the flags of the barriers, which are replaced. Memory writes read their address operand, which is
            // their "output".
            bool readByBarrier = false;
            bool readOtherwise = false;
            for(const auto& user : loc->getUsers())
            {
                auto mem = dynamic_cast<const MemoryInstruction*>(user.first);
                if(isBarrier(user.first) && user.second.readsLocal())
                    readByBarrier = true;
                else if(user.second.readsLocal() || (mem && mem->op != MemoryOperation::READ))
                    readOtherwise = true;
            }
            if(readByBarrier && !readOtherwise)
                continue;
            if(!loc->type.isScalarType() && !loc->type.getPointerType())
                return "keeps the non-scalar value " + loc->to_string() + " across a barrier";
            if(loc->type.getScalarBitCount() > 32)
                return "keeps the 64-bit value " + loc->to_string() + " across a barrier";
            info.contextValues.push_back(loc);
        }
        if(info.contextValues.size() > MAX_CONTEXT_VALUES)
            return "keeps too many values across a barrier (" + std::to_string(info.contextValues.size()) + ")";
        // deterministic order
        std::sort(info.contextValues.begin(), info.contextValues.end(),
            [](const Local* a, const Local* b) { return a->name < b->name; });
        barriers.push_back(std::move(info));
    }
    return "";
}

// Inserts a branch to the target if the condition is true, else to the next label
static void insertConditionalLoopBranch(Method& method, InstructionWalker& it, const Value& index,
    const Value& limit, const Local* loopTarget, const Local* exitTarget)
{
    auto cond = assignNop(it) = as_signed{index} < as_signed{limit};
    auto flag = assign(it, TYPE_BOOL, "%work_item_loop_continue") = (BOOL_TRUE, cond);
    assign(it, flag) = (BOOL_FALSE, cond.invert());
    BranchCond branchCond = BRANCH_ALWAYS;
    std::tie(it, branchCond) = insertBranchCondition(method, it, flag);
    it.emplace(std::make_unique<Branch>(loopTarget, branchCond));
    it.nextInBlock();
    it.emplace(std::make_unique<Branch>(exitTarget, branchCond.invert()));
    it.nextInBlock();
}

/*
 * The per-QPU loop state:
 * - the index of the QPU within the QPUs running the work-group (its first local linear ID) and their number,
 * - the local linear ID of the current work-item and the work-group size (the loop runs while the ID is smaller),
 * - the index of the current work-item on the QPU (the lane of the vectors keeping the values per work-item),
 * - the packed local IDs (x | y << 8 | z << 16) of the QPU's work-items, one per lane.
 */
struct LoopState
{
    Value qpuIndex = UNDEFINED_VALUE;
    Value numQPUs = UNDEFINED_VALUE;
    Value linearId = UNDEFINED_VALUE;
    Value groupSize = UNDEFINED_VALUE;
    Value lane = UNDEFINED_VALUE;
    Value localIds = UNDEFINED_VALUE;
    const Local* currentLocalIds = nullptr;
};

static LoopState insertLoopSetup(Method& method, InstructionWalker& it)
{
    LoopState state;
    const auto vectorType = TYPE_INT32.toVectorType(NATIVE_VECTOR_SIZE);
    // the local IDs (of the QPU's first work-item) and sizes as passed by the run-time
    auto uniformIds = method.findOrCreateBuiltin(BuiltinLocal::Type::LOCAL_IDS)->createReference();
    auto sizes = method.findOrCreateBuiltin(BuiltinLocal::Type::LOCAL_SIZES)->createReference();
    auto byteMask = Value(Literal(0xFFu), TYPE_INT32);
    // extracts the byte of the packed IDs/sizes for the dimension
    auto extractByte = [&](const Value& packed, uint32_t dimension, const std::string& name) -> Value {
        if(dimension == 0)
            return assign(it, TYPE_INT32, std::string(name)) = packed & byteMask;
        auto shifted = assign(it, TYPE_INT32, std::string(name)) =
            as_unsigned{packed} >> Value(Literal(dimension * 8u), TYPE_INT32);
        return assign(it, TYPE_INT32, std::string(name)) = shifted & byteMask;
    };
    auto sizeX = extractByte(sizes, 0, "%work_item_loop_size_x");
    auto sizeY = extractByte(sizes, 1, "%work_item_loop_size_y");
    auto sizeZ = extractByte(sizes, 2, "%work_item_loop_size_z");
    auto idX = extractByte(uniformIds, 0, "%work_item_loop_id_x");
    auto idY = extractByte(uniformIds, 1, "%work_item_loop_id_y");
    auto idZ = extractByte(uniformIds, 2, "%work_item_loop_id_z");
    auto sizeXY = assign(it, TYPE_INT32, "%work_item_loop_size_xy") = mul24(sizeX, sizeY);
    state.groupSize = assign(it, TYPE_INT32, "%work_item_loop_group_size") = mul24(sizeXY, sizeZ);
    // the QPU runs the work-items qpuIndex + k * numQPUs, the run-time starts QPU i with the local ID of work-item i
    auto offsetZ = assign(it, TYPE_INT32, "%work_item_loop_qpu_index") = mul24(idZ, sizeXY);
    auto offsetY = assign(it, TYPE_INT32, "%work_item_loop_qpu_index") = mul24(idY, sizeX);
    auto tmp = assign(it, TYPE_INT32, "%work_item_loop_qpu_index") = offsetZ + offsetY;
    state.qpuIndex = assign(it, TYPE_INT32, "%work_item_loop_qpu_index") = tmp + idX;
    state.numQPUs = assign(it, TYPE_INT32, "%work_item_loop_num_qpus") =
        min(as_signed{state.groupSize}, as_signed{Value(Literal(static_cast<uint32_t>(NUM_QPUS)), TYPE_INT32)});

    // the local IDs of the QPU's work-items, lane k for work-item qpuIndex + k * numQPUs. All values are below 256, so
    // 16-bit types allow the cheaper division via floating-point
    const auto shortVectorType = TYPE_INT16.toVectorType(NATIVE_VECTOR_SIZE);
    auto laneOffsets = assign(it, vectorType, "%work_item_loop_ids") = mul24(ELEMENT_NUMBER_REGISTER, state.numQPUs);
    auto linearIds = assign(it, shortVectorType, "%work_item_loop_ids") = laneOffsets + state.qpuIndex;
    auto shortSizeX = assign(it, TYPE_INT16, "%work_item_loop_size_x") = sizeX;
    auto shortSizeY = assign(it, TYPE_INT16, "%work_item_loop_size_y") = sizeY;
    auto quotientX = method.addNewLocal(shortVectorType, "%work_item_loop_ids");
    it.emplace(std::make_unique<IntrinsicOperation>("udiv", Value(quotientX), Value(linearIds), Value(shortSizeX)));
    it.nextInBlock();
    auto product = assign(it, vectorType, "%work_item_loop_ids") = mul24(quotientX, sizeX);
    auto localX = assign(it, vectorType, "%work_item_loop_ids") = linearIds - product;
    auto localZ = method.addNewLocal(shortVectorType, "%work_item_loop_ids");
    it.emplace(std::make_unique<IntrinsicOperation>("udiv", Value(localZ), Value(quotientX), Value(shortSizeY)));
    it.nextInBlock();
    product = assign(it, vectorType, "%work_item_loop_ids") = mul24(localZ, sizeY);
    auto localY = assign(it, vectorType, "%work_item_loop_ids") = quotientX - product;
    auto shiftedY = assign(it, vectorType, "%work_item_loop_ids") = localY << 8_val;
    auto shiftedZ = assign(it, vectorType, "%work_item_loop_ids") = localZ << 16_val;
    tmp = assign(it, vectorType, "%work_item_loop_ids") = localX | shiftedY;
    state.localIds = assign(it, vectorType, "%work_item_loop_ids") = tmp | shiftedZ;

    state.linearId = method.addNewLocal(TYPE_INT32, "%work_item_loop_linear_id");
    state.lane = method.addNewLocal(TYPE_INT32, "%work_item_loop_lane");
    state.currentLocalIds = method.addNewLocal(TYPE_INT32, "%work_item_loop_local_ids").local();
    return state;
}

static void insertLoopStart(InstructionWalker& it, const LoopState& state)
{
    assign(it, state.linearId) = state.qpuIndex;
    assign(it, state.lane) = INT_ZERO;
}

// Restores the local IDs and the given values of the current work-item
static void insertRestore(Method& method, InstructionWalker& it, const LoopState& state,
    const std::vector<const Local*>& values, const FastMap<const Local*, Value>& contexts)
{
    auto ids = method.addNewLocal(TYPE_INT32, "%work_item_loop_local_ids");
    it = insertVectorExtraction(it, method, state.localIds, state.lane, ids);
    it = insertReplication(it, ids, state.currentLocalIds->createReference());
    for(auto loc : values)
    {
        auto tmp = method.addNewLocal(TYPE_INT32, "%work_item_loop_restore");
        it = insertVectorExtraction(it, method, contexts.at(loc), state.lane, tmp);
        auto replicated = method.addNewLocal(TYPE_INT32, "%work_item_loop_restore");
        it = insertReplication(it, tmp, replicated);
        assign(it, loc->createReference()) = replicated;
    }
}

// Saves the given values of the current work-item and advances to the next work-item
static void insertSaveAndAdvance(Method& method, InstructionWalker& it, const LoopState& state,
    const std::vector<const Local*>& values, const FastMap<const Local*, Value>& contexts)
{
    // The saves use a copy of the lane, since VC4C may move the increment of the lane (a local written several times)
    // above earlier reads of it
    auto lane = assign(it, TYPE_INT32, "%work_item_loop_current_lane") = state.lane;
    for(auto loc : values)
    {
        auto tmp = assign(it, TYPE_INT32, "%work_item_loop_save") = loc->createReference();
        it = insertVectorInsertion(it, method, contexts.at(loc), lane, tmp);
    }
    auto nextLinearId = assign(it, TYPE_INT32, "%work_item_loop_next_linear_id") = state.linearId + state.numQPUs;
    auto nextLane = assign(it, TYPE_INT32, "%work_item_loop_next_lane") = lane + INT_ONE;
    assign(it, state.linearId) = nextLinearId;
    assign(it, state.lane) = nextLane;
}

static const Local* createLabel(Method& method, const std::string& name)
{
    return method.addNewLocal(TYPE_LABEL, name).local();
}

/*
 * The regions of a kernel are the parts executed between two barriers: region 0 starts at the start of the kernel,
 * region i (1 <= i <= number of barriers) right after barrier i. A region ends at the next barrier reached or at the
 * end of the kernel. With barriers in loops or (uniform) branches, a region can end at several barriers (e.g. at the
 * same barrier again for the next iteration, or at a later one after the loop), and a barrier can end several
 * regions. Since barriers are in work-group uniform control flow, all work-items of a QPU take the same path, i.e. end
 * the region at the same barrier.
 *
 * Returns for every barrier (index 1 to N) and the end of the kernel (index 0) the regions ending there.
 */
static std::vector<std::vector<std::size_t>> determineRegionEnds(Method& method, const std::vector<BarrierInfo>& barriers)
{
    FastMap<const BasicBlock*, std::size_t> barrierIndices;
    for(std::size_t i = 0; i < barriers.size(); ++i)
        barrierIndices.emplace(barriers[i].block, i + 1);
    std::vector<std::vector<std::size_t>> regionEnds(barriers.size() + 1);
    for(std::size_t region = 0; region <= barriers.size(); ++region)
    {
        // the region starts at the start of the kernel or with the rest of the barrier's block
        BasicBlock* start = region == 0 ? &*method.begin() : barriers[region - 1].block;
        FastSet<std::size_t> ends;
        std::vector<BasicBlock*> pending;
        if(endsKernel(*start))
            ends.emplace(0);
        for(auto next : getSuccessors(*start))
            pending.push_back(next);
        FastSet<BasicBlock*> visited;
        while(!pending.empty())
        {
            auto current = pending.back();
            pending.pop_back();
            if(!visited.emplace(current).second)
                continue;
            auto barrierIt = barrierIndices.find(current);
            if(barrierIt != barrierIndices.end())
            {
                ends.emplace(barrierIt->second);
                continue;
            }
            if(endsKernel(*current))
                ends.emplace(0);
            for(auto next : getSuccessors(*current))
                pending.push_back(next);
        }
        for(auto end : ends)
            regionEnds[end].push_back(region);
    }
    for(auto& regions : regionEnds)
        std::sort(regions.begin(), regions.end());
    return regionEnds;
}

/*
 * Inserts the end of a region for the current work-item: saves its values, advances to the next work-item and
 * continues with the region the QPU is in for that work-item, or (after the last work-item) continues at the exit
 * label.
 */
static void insertRegionEnd(Method& method, InstructionWalker& it, const LoopState& state,
    const std::vector<const Local*>& values, const FastMap<const Local*, Value>& contexts,
    const std::vector<std::size_t>& regions, const std::vector<const Local*>& headers, const Value& currentRegion,
    const Local* exitLabel)
{
    insertSaveAndAdvance(method, it, state, values, contexts);
    if(regions.empty())
    {
        // not reachable
        it.emplace(std::make_unique<Branch>(exitLabel));
        it.nextInBlock();
        return;
    }
    if(regions.size() == 1)
    {
        insertConditionalLoopBranch(method, it, state.linearId, state.groupSize, headers[regions.front()], exitLabel);
        return;
    }
    // the region ending here is only known at run-time
    auto dispatchLabel = createLabel(method, "%work_item_loop_dispatch");
    insertConditionalLoopBranch(method, it, state.linearId, state.groupSize, dispatchLabel, exitLabel);
    it = method.emplaceLabel(it, std::make_unique<BranchLabel>(*dispatchLabel));
    it.nextInBlock();
    for(std::size_t i = 0; i + 1 < regions.size(); ++i)
    {
        auto cond = assignNop(it) =
            as_signed{currentRegion} == as_signed{Value(Literal(static_cast<uint32_t>(regions[i])), TYPE_INT32)};
        auto flag = assign(it, TYPE_BOOL, "%work_item_loop_region") = (BOOL_TRUE, cond);
        assign(it, flag) = (BOOL_FALSE, cond.invert());
        BranchCond branchCond = BRANCH_ALWAYS;
        std::tie(it, branchCond) = insertBranchCondition(method, it, flag);
        it.emplace(std::make_unique<Branch>(headers[regions[i]], branchCond));
        it.nextInBlock();
        auto nextLabel = createLabel(method, "%work_item_loop_dispatch");
        it.emplace(std::make_unique<Branch>(nextLabel, branchCond.invert()));
        it.nextInBlock();
        it = method.emplaceLabel(it, std::make_unique<BranchLabel>(*nextLabel));
        it.nextInBlock();
    }
    it.emplace(std::make_unique<Branch>(headers[regions.back()]));
    it.nextInBlock();
}

static void convertToWorkItemLoops(Method& method, const std::vector<BarrierInfo>& barriers)
{
    const auto contextType = TYPE_INT32.toVectorType(NATIVE_VECTOR_SIZE);
    FastMap<const Local*, Value> contexts;
    for(const auto& barrier : barriers)
    {
        for(auto loc : barrier.contextValues)
        {
            if(contexts.find(loc) == contexts.end())
                contexts.emplace(loc, method.addNewLocal(contextType, "%work_item_loop_context"));
        }
    }
    // determined before changing the control flow
    auto regionEnds = determineRegionEnds(method, barriers);
    std::vector<const Local*> headers;
    for(std::size_t region = 0; region <= barriers.size(); ++region)
        headers.push_back(createLabel(method, "%work_item_loop_header"));
    // the region the QPU is in, the same for all of its work-items
    auto currentRegion = method.addNewLocal(TYPE_INT32, "%work_item_loop_region");

    // the start of the kernel: loop setup and the start of the first region
    auto it = method.begin()->walk().nextInBlock();
    auto state = insertLoopSetup(method, it);
    for(const auto& context : contexts)
        // the lanes are only read after being written, but the registers are initialized as a whole
        assign(it, context.second) = INT_ZERO;
    insertLoopStart(it, state);
    it = method.emplaceLabel(it, std::make_unique<BranchLabel>(*headers[0]));
    it.nextInBlock();
    assign(it, currentRegion) = INT_ZERO;
    insertRestore(method, it, state, {}, contexts);

    // the end of the regions at the barriers, and the start of the regions after them
    for(std::size_t i = 0; i < barriers.size(); ++i)
    {
        const auto& barrier = barriers[i];
        it = barrier.block->walk().nextInBlock();
        // it points to the barrier call
        auto syncLabel = createLabel(method, "%work_item_loop_sync");
        insertRegionEnd(method, it, state, barrier.contextValues, contexts, regionEnds[i + 1], headers, currentRegion,
            syncLabel);
        it = method.emplaceLabel(it, std::make_unique<BranchLabel>(*syncLabel));
        it.nextInBlock();
        // replace the barrier between the work-items with the one between the QPUs of the work-group
        it.reset(std::make_unique<MethodCall>(
            std::string(WORK_ITEM_LOOP_BARRIER_FUNCTION), std::vector<Value>{state.qpuIndex, state.numQPUs}));
        it.nextInBlock();
        insertLoopStart(it, state);
        it = method.emplaceLabel(it, std::make_unique<BranchLabel>(*headers[i + 1]));
        it.nextInBlock();
        assign(it, currentRegion) = Value(Literal(static_cast<uint32_t>(i + 1)), TYPE_INT32);
        insertRestore(method, it, state, barrier.contextValues, contexts);
    }

    // the end of the kernel: all returns go to the end of the last region
    auto latchLabel = createLabel(method, "%work_item_loop_end");
    for(auto& block : method)
    {
        for(auto blockIt = block.walk(); !blockIt.isEndOfBlock(); blockIt.nextInBlock())
        {
            if(blockIt.has() && blockIt.get<Return>())
                blockIt.reset(std::make_unique<Branch>(latchLabel));
        }
    }
    auto lastBlockIt = method.begin();
    for(auto blockIt = method.begin(); blockIt != method.end(); ++blockIt)
        lastBlockIt = blockIt;
    it = method.emplaceLabel(lastBlockIt->walkEnd(), std::make_unique<BranchLabel>(*latchLabel));
    it.nextInBlock();
    auto exitLabel = createLabel(method, "%work_item_loop_exit");
    insertRegionEnd(method, it, state, {}, contexts, regionEnds[0], headers, currentRegion, exitLabel);
    it = method.emplaceLabel(it, std::make_unique<BranchLabel>(*exitLabel));
    it.nextInBlock();
    it.emplace(std::make_unique<Return>());

    method.metaData.workItemLoopLocalIds = state.currentLocalIds;
}

void normalization::loopWorkItems(Module& module, Method& method, const Configuration& config)
{
    if(!isWorkItemLoopsEnabled(config))
        return;
    std::vector<BarrierInfo> barriers;
    auto reason = checkKernel(method, barriers);
    if(!reason.empty())
    {
        if(reason != "has no barriers")
            CPPLOG_LAZY(logging::Level::INFO,
                log << "Work-item loops: not used for kernel '" << method.name << "': " << reason << logging::endl);
        return;
    }
    convertToWorkItemLoops(method, barriers);
    CPPLOG_LAZY(logging::Level::INFO,
        log << "Work-item loops: kernel '" << method.name << "' runs up to " << MAX_WORK_ITEMS_PER_QPU
            << " work-items per QPU (" << barriers.size() << " barriers)" << logging::endl);
    method.dumpInstructions();
}
