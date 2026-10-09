/*
 * See the file "LICENSE" for the full license governing this code.
 */

#include "SIMT.h"

#include "../InstructionWalker.h"
#include "../Method.h"
#include "../Module.h"
#include "../intermediate/Helper.h"
#include "../intermediate/IntermediateInstruction.h"
#include "../intermediate/VectorHelper.h"
#include "../intermediate/operators.h"
#include "../intrinsics/WorkItems.h"
#include "WorkItemLoops.h"
#include "log.h"

#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace vc4c;
using namespace vc4c::intermediate;
using namespace vc4c::operators;

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
        // Whether the value in lane i is 16 * u + i for some u >= 0 (the same for all lanes), e.g. the local ID in x
        // (the chunks of work-items start at multiples of 16). Then dividing it by a multiple of 16 gives the same
        // value in all lanes, e.g. for the index of the row in tid / 16 with tid = get_local_id(0) + 16 *
        // get_local_id(1).
        bool aligned = false;

        bool operator==(const LaneInfo& other) const
        {
            return stride == other.stride && aligned == other.aligned;
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

        bool isAligned(const Value& val) const
        {
            if(auto loc = val.checkLocal())
            {
                auto it = varying.find(loc);
                return it != varying.end() && it->second.aligned;
            }
            return false;
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

/*
 * Divergent control flow
 *
 * A conditional branch on a work-item dependent value can go different ways for the lanes of a QPU. Such a branch and
 * the blocks up to its merge point (the immediate post-dominator) form a divergent region. The region's blocks run one
 * after the other, each only for the lanes reaching it (the block's mask), and are skipped if no lane does:
 * - values used outside of the block are only written for the block's lanes (written to a temporary, which is moved
 *   into the value at the end of the block for the active lanes only),
 * - stores only write the block's lanes (see insertMaskedStore),
 * - loads need no change: all addresses are contiguous across the lanes, so the load of an inactive lane reads next to
 *   the address of an active lane.
 * Only regions without loops are supported, see doc/SIMT.md.
 */
namespace
{
    enum class EdgeKind
    {
        // taken by all lanes reaching the end of the block
        ALWAYS,
        // taken by the lanes where the condition is true
        ON_TRUE,
        // taken by the lanes where the condition is false
        ON_FALSE
    };

    struct Edge
    {
        BasicBlock* target;
        EdgeKind kind;
    };

    struct BlockExit
    {
        std::vector<Edge> edges;
        // the condition of the conditional edges
        Optional<Value> condition;
        // the instructions implementing the exit: the instruction setting the branch flags and the branches
        std::vector<InstructionWalker> instructions;
        // why the exit is not supported, if it isn't
        std::string error;
    };

    struct ControlFlow
    {
        // in the order of the method
        std::vector<BasicBlock*> blocks;
        FastMap<const BasicBlock*, std::size_t> indices;
        std::vector<BlockExit> exits;
        // the immediate post-dominator of every block, blocks.size() for the (virtual) exit of the method
        std::vector<std::size_t> postDominators;
        // whether a block (second index) post-dominates another one (first index)
        std::vector<std::vector<bool>> postDominatorSets;
    };

    struct Region
    {
        // the block before the region, run by all lanes: the block with the divergent branch, or the block before a
        // loop the lanes may leave in different iterations
        std::size_t entryBlock;
        // the first post-dominator of the entry block after the region, where all lanes run again
        std::size_t mergeBlock;
        // all blocks between the entry and merge blocks, in order
        std::vector<std::size_t> blocks;
        // the loops in the region as pairs of header and latch (the last block of the loop with the only back edge)
        std::vector<std::pair<std::size_t, std::size_t>> loops;
    };

    using DivergencePredicate = std::function<bool(const Value& condition)>;
} // namespace

/*
 * Scalars of at most 32 bits, and in vector kernels (see determineVectorWidth) vectors of the kernel's vector width with
 * 32-bit elements.
 */
static bool isSupportedType(const DataType& type, uint8_t vectorWidth)
{
    if(type.getPointerType())
        return true;
    if(type.isLabelType() || type.isVoidType())
        return true;
    if(type.isScalarType())
        return type.getScalarBitCount() <= 32;
    return vectorWidth > 1 && type.getVectorWidth() == vectorWidth && type.getScalarBitCount() == 32;
}

/*
 * The vector width of the kernel: 1 if it uses no vector types, the single vector width N (2, 4 or 8) of all its
 * vectors otherwise. Such a kernel runs with 16 / N work-items per QPU, each in N adjacent SIMD lanes: lane
 * (work-item * N + element). Returns 0 if the vector types are not supported.
 *
 * Kernels using only 16-element vectors already use all SIMD lanes, so they don't run in SIMT mode.
 */
static uint8_t determineVectorWidth(const Method& method, std::string& reason)
{
    std::set<unsigned> widths;
    const IntermediateInstruction* currentInstruction = nullptr;
    auto addType = [&](const DataType& type) {
        auto ptrType = type.getPointerType();
        const auto& valueType = ptrType ? ptrType->elementType : type;
        if(!valueType.isLabelType() && valueType.isVectorType() && widths.emplace(valueType.getVectorWidth()).second)
            reason += " " + valueType.to_string() + " (" +
                (currentInstruction ? currentInstruction->to_string() : std::string("parameter")) + ")";
    };
    for(const auto& param : method.parameters)
        addType(param.type);
    for(const auto& block : method)
    {
        for(const auto& instr : block)
        {
            if(!instr)
                continue;
            currentInstruction = instr.get();
            instr->forUsedLocals(
                [&](const Local* loc, LocalUse::Type, const IntermediateInstruction&) { addType(loc->type); });
            for(const auto& arg : instr->getArguments())
                addType(arg.type);
        }
    }
    // the element number register is a 16-element vector, but no vector of the kernel. Kernels using only 16-element
    // vectors don't run in SIMT mode either, see isSupportedType.
    widths.erase(NATIVE_VECTOR_SIZE);
    if(widths.empty())
        return 1;
    if(widths.size() == 1 && (*widths.begin() == 2 || *widths.begin() == 4 || *widths.begin() == 8))
        return static_cast<uint8_t>(*widths.begin());
    return 0;
}

static bool isWorkItemFunction(const MethodCall& call)
{
    for(const auto& name : {intrinsics::FUNCTION_NAME_LOCAL_SIZE, intrinsics::FUNCTION_NAME_LOCAL_ID,
            intrinsics::FUNCTION_NAME_NUM_DIMENSIONS, intrinsics::FUNCTION_NAME_NUM_GROUPS,
            intrinsics::FUNCTION_NAME_GROUP_ID, intrinsics::FUNCTION_NAME_GLOBAL_OFFSET,
            intrinsics::FUNCTION_NAME_GLOBAL_SIZE, intrinsics::FUNCTION_NAME_GLOBAL_ID})
    {
        if(call.methodName == name)
            return true;
    }
    return false;
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

static bool isBarrier(const MethodCall& call)
{
    return call.methodName == "vc4cl_barrier";
}

/*
 * Rejects kernels using functionality which is not (yet) supported in SIMT mode. Returns the reason, or an empty
 * string if the kernel might be supported.
 *
 * Kernels with barriers or __local memory are supported if the QPUs can loop over the chunks of work-items of their
 * work-group (see normalization/WorkItemLoops.cpp): then the run-time runs a single work-group at a time, and a barrier
 * synchronizes the QPUs after all chunks reached it.
 */
static std::string checkUnsupported(const Method& method, uint8_t vectorWidth, bool allowWorkGroupSync)
{
    // Any dimensions work, since the run-time deals chunks of up to 16 work-items of a row (same local IDs in y and
    // z) to the QPUs
    const auto& sizes = method.metaData.workGroupSizes;
    if(sizes[0] != 0 &&
        static_cast<uint32_t>(sizes[0]) * std::max(sizes[1], uint32_t{1}) * std::max(sizes[2], uint32_t{1}) >
            NUM_QPUS * SIMT_WIDTH)
        return "required work-group size is larger than 192";
    if(!method.stackAllocations.empty())
        return "uses private memory (stack allocations)";
    for(const auto& param : method.parameters)
    {
        if(!isSupportedType(param.type, vectorWidth) || (vectorWidth > 1 && param.type.isVectorType()))
            return "parameter type " + param.type.to_string();
        if(auto ptrType = param.type.getPointerType())
        {
            if(ptrType->addressSpace == AddressSpace::LOCAL && !allowWorkGroupSync)
                return "uses __local memory";
        }
    }

    for(const auto& block : method)
    {
        for(const auto& instr : block)
        {
            if(!instr)
                continue;
            if(dynamic_cast<const MemoryBarrier*>(instr.get()) ||
                dynamic_cast<const SemaphoreAdjustment*>(instr.get()) || dynamic_cast<const MutexLock*>(instr.get()))
                return "uses synchronization: " + instr->to_string();
            if(auto call = dynamic_cast<const MethodCall*>(instr.get()))
            {
                if(allowWorkGroupSync && isBarrier(*call))
                    continue;
                // printf() writes one record per QPU, see LowerPrintf
                for(const char* unsupported : {"barrier", "atomic", "mutex", "semaphore", "dma", "vpm", "fence", "async",
                        "prefetch", "linear_id", "printf"})
                {
                    if(call->methodName.find(unsupported) != std::string::npos)
                        return "calls " + call->methodName;
                }
                if(isWorkItemIdCall(*call) && !getDimension(*call))
                    return "work-item ID with non-constant dimension";
                // the functions are intrinsified after this pass and wouldn't know about the vector layout
                if(vectorWidth > 1 && !isWorkItemFunction(*call))
                    return "calls " + call->methodName + " in a vector kernel";
            }
            if(vectorWidth > 1)
            {
                // vector constants other than splats would need to be repeated for every work-item
                auto load = dynamic_cast<const LoadImmediate*>(instr.get());
                if(load && load->type != LoadType::REPLICATE_INT32)
                    return "vector constant in a vector kernel: " + instr->to_string();
                for(const auto& arg : instr->getArguments())
                {
                    if(arg.type.isVectorType() && !arg.checkLocal() && !arg.checkRegister() && !arg.isAllSame())
                        return "vector constant in a vector kernel: " + instr->to_string();
                }
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
                if(!isSupportedType(loc->type, vectorWidth))
                    unsupportedType = true;
            });
            if(unsupportedType)
                return "uses vector or 64-bit values: " + instr->to_string();
            bool usesLocalMemory = false;
            // also finds the __local variables of the kernel, which are globals. The work-groups running at the same
            // time would share them.
            instr->forUsedLocals([&](const Local* loc, LocalUse::Type, const IntermediateInstruction&) {
                auto ptrType = loc->type.getPointerType();
                if(ptrType && ptrType->addressSpace == AddressSpace::LOCAL)
                    usesLocalMemory = true;
            });
            if(usesLocalMemory && !allowWorkGroupSync)
                return "uses __local memory: " + instr->to_string();
        }
    }
    return "";
}

// The next block is the block following in the method, if any
static BlockExit determineExit(Method& method, BasicBlock& block, BasicBlock* nextBlock)
{
    BlockExit exit;
    for(const auto& instr : block)
    {
        if(dynamic_cast<const Return*>(instr.get()))
            // the end of the kernel
            return exit;
    }

    // the branches at the end of the block
    auto firstBranch = block.walkEnd();
    for(auto it = block.walkEnd(); !it.isStartOfBlock();)
    {
        it.previousInBlock();
        if(!it.has() || !it.get<Branch>())
            break;
        firstBranch = it;
    }

    bool hasTrue = false;
    bool hasFalse = false;
    bool hasUnconditional = false;
    for(auto it = firstBranch.copy(); !it.isEndOfBlock() && !hasUnconditional; it.nextInBlock())
    {
        auto branch = it.get<Branch>();
        if(!branch)
            continue;
        auto target = branch->getSingleTargetLabel();
        auto targetBlock = target ? method.findBasicBlock(target) : nullptr;
        if(!targetBlock)
        {
            exit.error = "unsupported branch: " + branch->to_string();
            return exit;
        }
        exit.instructions.push_back(it);
        if(branch->isUnconditional())
        {
            // after a single conditional branch, this is the other direction
            auto kind = hasTrue ? EdgeKind::ON_FALSE : (hasFalse ? EdgeKind::ON_TRUE : EdgeKind::ALWAYS);
            if(!(hasTrue && hasFalse))
                exit.edges.push_back(Edge{targetBlock, kind});
            hasUnconditional = true;
        }
        else if(branch->branchCondition == BRANCH_ALL_Z_CLEAR && !hasTrue)
        {
            // as inserted by insertBranchCondition: taken if the condition is true
            exit.edges.push_back(Edge{targetBlock, EdgeKind::ON_TRUE});
            hasTrue = true;
        }
        else if(branch->branchCondition == BRANCH_ANY_Z_SET && !hasFalse)
        {
            exit.edges.push_back(Edge{targetBlock, EdgeKind::ON_FALSE});
            hasFalse = true;
        }
        else
        {
            exit.error = "unsupported branch condition: " + branch->to_string();
            return exit;
        }
    }
    if(!hasUnconditional && !(hasTrue && hasFalse))
    {
        // falls through to the next block
        if(auto next = nextBlock)
            exit.edges.push_back(
                Edge{next, hasTrue ? EdgeKind::ON_FALSE : (hasFalse ? EdgeKind::ON_TRUE : EdgeKind::ALWAYS)});
    }

    if(hasTrue || hasFalse)
    {
        // the flags are set by the last instruction setting flags before the branches
        auto setter = firstBranch.copy();
        while(!setter.isStartOfBlock())
        {
            setter.previousInBlock();
            if(setter.has() && setter->doesSetFlag())
                break;
        }
        auto condition = setter.has() ? getBranchCondition(setter.get<ExtendedInstruction>()) :
                                        std::make_pair(Optional<Value>{}, std::bitset<NATIVE_VECTOR_SIZE>{});
        if(!condition.first || condition.second != 0x1)
        {
            exit.error = "unsupported branch condition in block " + block.to_string();
            return exit;
        }
        exit.condition = condition.first;
        exit.instructions.insert(exit.instructions.begin(), setter);
    }
    return exit;
}

static ControlFlow determineControlFlow(Method& method)
{
    ControlFlow cf;
    for(auto& block : method)
    {
        cf.indices.emplace(&block, cf.blocks.size());
        cf.blocks.push_back(&block);
    }
    for(std::size_t b = 0; b < cf.blocks.size(); ++b)
        cf.exits.push_back(
            determineExit(method, *cf.blocks[b], b + 1 < cf.blocks.size() ? cf.blocks[b + 1] : nullptr));

    // post-dominator sets, index blocks.size() is the virtual exit of the method
    auto numBlocks = cf.blocks.size();
    std::vector<std::vector<std::size_t>> successors(numBlocks);
    for(std::size_t b = 0; b < numBlocks; ++b)
    {
        for(const auto& edge : cf.exits[b].edges)
            successors[b].push_back(cf.indices.at(edge.target));
        if(successors[b].empty())
            successors[b].push_back(numBlocks);
    }
    std::vector<std::vector<bool>> postDominatorSets(numBlocks + 1, std::vector<bool>(numBlocks + 1, true));
    postDominatorSets[numBlocks] = std::vector<bool>(numBlocks + 1, false);
    postDominatorSets[numBlocks][numBlocks] = true;
    bool changed = true;
    while(changed)
    {
        changed = false;
        for(std::size_t b = numBlocks; b-- > 0;)
        {
            std::vector<bool> result(numBlocks + 1, true);
            for(auto succ : successors[b])
            {
                for(std::size_t k = 0; k <= numBlocks; ++k)
                    result[k] = result[k] && postDominatorSets[succ][k];
            }
            result[b] = true;
            if(result != postDominatorSets[b])
            {
                postDominatorSets[b] = std::move(result);
                changed = true;
            }
        }
    }
    // the immediate post-dominator is the closest one, i.e. the one with the most post-dominators itself
    cf.postDominators.assign(numBlocks, numBlocks);
    for(std::size_t b = 0; b < numBlocks; ++b)
    {
        std::size_t bestCount = 0;
        for(std::size_t p = 0; p <= numBlocks; ++p)
        {
            if(p == b || !postDominatorSets[b][p])
                continue;
            auto count = static_cast<std::size_t>(
                std::count(postDominatorSets[p].begin(), postDominatorSets[p].end(), true));
            if(count > bestCount)
            {
                bestCount = count;
                cf.postDominators[b] = p;
            }
        }
    }
    cf.postDominatorSets = std::move(postDominatorSets);
    return cf;
}

/*
 * Determines the divergent region of the given block with a divergent branch. Returns the reason if it is not
 * supported.
 *
 * The region contains all blocks between its entry block and its merge block, which must be exactly the blocks reached
 * from the entry before reaching the merge block. If the divergent branch is inside a loop, the lanes may leave the
 * loop in different iterations, so the region grows to contain the whole loop: its entry is then the block before the
 * loop header. Loops in the region need to consist of consecutive blocks with the header first and the only back edge from
 * the last block (the latch).
 */
static std::string determineRegion(const ControlFlow& cf, std::size_t branchBlock, Region& region)
{
    auto numBlocks = cf.blocks.size();
    const auto blockName = cf.blocks[branchBlock]->to_string();
    auto index = [&](const Edge& edge) -> std::size_t { return cf.indices.at(edge.target); };

    auto start = branchBlock;
    auto end = cf.postDominators[branchBlock];
    std::set<std::size_t> reachable;
    bool changed = true;
    for(std::size_t iteration = 0; changed; ++iteration)
    {
        changed = false;
        if(end == numBlocks || iteration > 2 * numBlocks)
            return "divergent branch without merge point in " + blockName;
        // all blocks reachable from the entry without passing the merge block
        reachable.clear();
        std::vector<std::size_t> worklist;
        for(const auto& edge : cf.exits[start].edges)
            worklist.push_back(index(edge));
        while(!worklist.empty())
        {
            auto b = worklist.back();
            worklist.pop_back();
            if(b == end || !reachable.emplace(b).second)
                continue;
            for(const auto& edge : cf.exits[b].edges)
                worklist.push_back(index(edge));
        }
        if(reachable.empty())
            return "empty divergent region in " + blockName;
        auto first = *reachable.begin();
        auto last = *reachable.rbegin();
        if(first <= start)
        {
            // inside a loop starting at the first block: start before the loop
            if(first == 0)
                return "divergent loop at the start of the kernel in " + blockName;
            start = first - 1;
            changed = true;
            continue;
        }
        if(last > end || end <= start || !cf.postDominatorSets[start][end])
        {
            // all lanes run again at the next post-dominator of the entry after all blocks of the region
            auto p = cf.postDominators[start];
            while(p != numBlocks && (p <= last || p <= start))
                p = cf.postDominators[p];
            if(p != end)
            {
                end = p;
                changed = true;
            }
        }
    }

    region.entryBlock = start;
    region.mergeBlock = end;
    region.blocks.clear();
    region.loops.clear();

    // the blocks run one after the other in their order, so they need to be exactly the blocks between the entry and
    // merge blocks
    for(std::size_t b = start + 1; b < end; ++b)
    {
        if(reachable.find(b) == reachable.end())
            return "divergent region of " + blockName + " contains unrelated " + cf.blocks[b]->to_string();
        region.blocks.push_back(b);
    }
    if(!cf.exits[start].error.empty())
        return cf.exits[start].error;
    for(auto b : reachable)
    {
        if(b <= start || b >= end)
            return "unsupported control flow in divergent region of " + blockName;
        if(!cf.exits[b].error.empty())
            return cf.exits[b].error;
    }
    // only entered from the entry block
    for(std::size_t b = 0; b < numBlocks; ++b)
    {
        if(b == start || reachable.find(b) != reachable.end())
            continue;
        for(const auto& edge : cf.exits[b].edges)
        {
            if(reachable.find(index(edge)) != reachable.end())
                return "divergent region of " + blockName + " is entered from " + cf.blocks[b]->to_string();
        }
    }

    // the loops: back edges go to the header from the last block of the loop
    std::map<std::size_t, std::size_t> latches;
    for(auto b : reachable)
    {
        for(const auto& edge : cf.exits[b].edges)
        {
            auto target = index(edge);
            if(target > b)
                continue;
            if(target <= start || !latches.emplace(target, b).second)
                return "loop with several back edges in divergent region of " + blockName;
        }
    }
    for(const auto& loop : latches)
    {
        auto header = loop.first;
        auto latch = loop.second;
        for(std::size_t b = start; b < end; ++b)
        {
            bool inLoop = b >= header && b <= latch;
            for(const auto& edge : cf.exits[b].edges)
            {
                auto target = index(edge);
                bool targetInLoop = target >= header && target <= latch;
                if(inLoop && target < header)
                    return "loop leaving to an earlier block in divergent region of " + blockName;
                if(inLoop && target == header && b != latch)
                    return "loop with several back edges in divergent region of " + blockName;
                if(!inLoop && targetInLoop && target != header)
                    return "loop entered in the middle in divergent region of " + blockName;
            }
        }
        for(const auto& other : latches)
        {
            // loops are either nested or disjoint
            bool overlaps = other.first <= latch && header <= other.second;
            bool nested = (other.first >= header && other.second <= latch) ||
                (header >= other.first && latch <= other.second);
            if(overlaps && !nested)
                return "overlapping loops in divergent region of " + blockName;
        }
        region.loops.emplace_back(header, latch);
    }
    return "";
}

/*
 * Determines the outermost divergent regions. Divergent branches inside a region are handled as part of it. Returns the
 * first reason why a region is not supported, if any. The returned regions are all supported.
 */
static std::string findDivergentRegions(
    const ControlFlow& cf, const DivergencePredicate& isDivergent, std::vector<Region>& regions)
{
    std::string error;
    regions.clear();
    for(std::size_t b = 0; b < cf.blocks.size(); ++b)
    {
        if(!regions.empty() && b < regions.back().mergeBlock)
            // inside the previous region
            continue;
        const auto& exit = cf.exits[b];
        if(!exit.condition || !isDivergent(*exit.condition))
            continue;
        Region region;
        auto reason = determineRegion(cf, b, region);
        if(reason.empty() && !regions.empty() && region.entryBlock < regions.back().mergeBlock)
            reason = "overlapping divergent regions at " + cf.blocks[b]->to_string();
        if(reason.empty())
            regions.push_back(std::move(region));
        else if(error.empty())
            error = reason;
    }
    return error;
}

// All blocks using a local
static FastMap<const Local*, FastSet<const BasicBlock*>> determineLocalBlocks(Method& method)
{
    FastMap<const Local*, FastSet<const BasicBlock*>> result;
    for(const auto& block : method)
    {
        for(const auto& instr : block)
        {
            if(!instr)
                continue;
            instr->forUsedLocals([&](const Local* loc, LocalUse::Type, const IntermediateInstruction&) {
                result[loc].emplace(&block);
            });
        }
    }
    return result;
}

static bool isUsedOutside(
    const FastMap<const Local*, FastSet<const BasicBlock*>>& localBlocks, const Local* loc, const BasicBlock& block)
{
    auto it = localBlocks.find(loc);
    return it != localBlocks.end() && (it->second.size() > 1 || it->second.find(&block) == it->second.end());
}

static bool isMemoryWrite(const IntermediateInstruction& instr)
{
    auto mem = dynamic_cast<const MemoryInstruction*>(&instr);
    return mem && mem->op == MemoryOperation::WRITE;
}

static bool writesReplicationRegister(const IntermediateInstruction& instr)
{
    auto out = instr.getOutput();
    return out && (out->hasRegister(REG_REPLICATE_ALL) || out->hasRegister(REG_REPLICATE_QUAD));
}

static bool readsReplicationRegister(const IntermediateInstruction& instr)
{
    for(const auto& arg : instr.getArguments())
    {
        if(arg.hasRegister(REG_REPLICATE_ALL) || arg.hasRegister(REG_REPLICATE_QUAD) || arg.hasRegister(REG_ACC5))
            return true;
    }
    return false;
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
        it->second.aligned = false;
        changed = true;
    }
}

static Optional<int64_t> scaleStride(Optional<int64_t> stride, int64_t factor)
{
    if(!stride)
        return {};
    return *stride * factor;
}

// The power of two (at least 16) of the literal, if it is one
static Optional<uint32_t> getMultipleOf16Shift(const Value& val)
{
    auto lit = val.getLiteralValue();
    if(!lit || lit->signedInt() < 16 || (lit->unsignedInt() & (lit->unsignedInt() - 1)) != 0)
        return {};
    uint32_t shift = 0;
    while((1u << shift) < lit->unsignedInt())
        ++shift;
    return shift;
}

/*
 * Whether the work-group uniform value is not negative (assuming no overflows), e.g. the work-item IDs and sizes and
 * values computed from them by additions, multiplications and shifts.
 */
static bool isNonNegativeUniform(const Analysis& analysis, const Value& val, unsigned depth = 0)
{
    if(analysis.isVarying(val) || depth > 8)
        return false;
    if(auto lit = val.getLiteralValue())
        return lit->signedInt() >= 0;
    auto loc = val.checkLocal();
    auto writer = loc ? loc->getSingleWriter() : nullptr;
    if(!writer || writer->hasConditionalExecution() || writer->hasUnpackMode() || writer->hasPackMode())
        return false;
    if(auto call = dynamic_cast<const MethodCall*>(writer))
        return isWorkItemFunction(*call);
    if(auto move = dynamic_cast<const MoveOperation*>(writer))
        return !move->getVectorRotation() && isNonNegativeUniform(analysis, move->getSource(), depth + 1);
    auto op = dynamic_cast<const Operation*>(writer);
    auto intrinsic = dynamic_cast<const IntrinsicOperation*>(writer);
    bool isSupported = (op && (op->op == OP_ADD || op->op == OP_SHL || op->op == OP_SHR || op->op == OP_AND)) ||
        (intrinsic && (intrinsic->opCode == "mul" || intrinsic->opCode == "zext" || intrinsic->opCode == "udiv"));
    if(!isSupported)
        return false;
    for(const auto& arg : writer->getArguments())
    {
        if(!isNonNegativeUniform(analysis, arg, depth + 1))
            return false;
    }
    return true;
}

// Whether the work-group uniform value is a non-negative multiple of 16 (assuming no overflows)
static bool isUniformMultipleOf16(const Analysis& analysis, const Value& val, unsigned depth = 0)
{
    if(analysis.isVarying(val) || depth > 8)
        return false;
    if(auto lit = val.getLiteralValue())
        return lit->signedInt() >= 0 && lit->unsignedInt() % 16 == 0;
    auto loc = val.checkLocal();
    auto writer = loc ? loc->getSingleWriter() : nullptr;
    if(!writer || writer->hasConditionalExecution() || writer->hasUnpackMode() || writer->hasPackMode())
        return false;
    auto arg0 = writer->getArgument(0);
    auto arg1 = writer->getArgument(1);
    if(auto move = dynamic_cast<const MoveOperation*>(writer))
        return !move->getVectorRotation() && isUniformMultipleOf16(analysis, move->getSource(), depth + 1);
    if(auto op = dynamic_cast<const Operation*>(writer))
    {
        if(op->op == OP_SHL && arg0 && arg1)
        {
            auto shift = arg1->getLiteralValue();
            return shift && shift->signedInt() >= 4 && shift->signedInt() < 31 &&
                isNonNegativeUniform(analysis, *arg0, depth + 1);
        }
        if(op->op == OP_ADD && arg0 && arg1)
            return isUniformMultipleOf16(analysis, *arg0, depth + 1) &&
                isUniformMultipleOf16(analysis, *arg1, depth + 1);
        return false;
    }
    auto intrinsic = dynamic_cast<const IntrinsicOperation*>(writer);
    if(intrinsic && intrinsic->opCode == "mul" && arg0 && arg1)
        return (isUniformMultipleOf16(analysis, *arg0, depth + 1) && isNonNegativeUniform(analysis, *arg1, depth + 1)) ||
            (isUniformMultipleOf16(analysis, *arg1, depth + 1) && isNonNegativeUniform(analysis, *arg0, depth + 1));
    return false;
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
    {
        // values computed only from values which are the same in all lanes (stride 0, e.g. tid / 16, see LaneInfo)
        // and uniform values are the same in all lanes too
        auto move = dynamic_cast<const MoveOperation*>(&instr);
        bool sameInAllLanes = arg0 && !(move && move->getVectorRotation()) &&
            (dynamic_cast<const Operation*>(&instr) || dynamic_cast<const IntrinsicOperation*>(&instr) || move);
        for(const auto& arg : instr.getArguments())
        {
            if(arg.checkRegister())
                // e.g. the element number
                sameInAllLanes = false;
            else if(analysis.isVarying(arg) && analysis.getStride(arg) != int64_t{0})
                sameInAllLanes = false;
        }
        if(sameInAllLanes)
            return LaneInfo{int64_t{0}};
    }
    if(auto move = dynamic_cast<const MoveOperation*>(&instr))
    {
        if(!move->getVectorRotation() && !instr.hasUnpackMode() && !instr.hasPackMode())
            return LaneInfo{analysis.getStride(move->getSource()), analysis.isAligned(move->getSource())};
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
            bool aligned = (analysis.isAligned(*arg0) && isUniformMultipleOf16(analysis, *arg1)) ||
                (analysis.isAligned(*arg1) && isUniformMultipleOf16(analysis, *arg0));
            if(s0 && s1)
                return LaneInfo{*s0 + *s1, aligned};
        }
        if((op->op == OP_SHR || op->op == OP_ASR) && arg1 && analysis.isAligned(*arg0))
        {
            // (16 * u + i) >> k is u >> (k - 4) in all lanes
            auto shift = arg1->getLiteralValue();
            if(shift && !analysis.isVarying(*arg1) && shift->signedInt() >= 4 && shift->signedInt() < 32)
                return LaneInfo{int64_t{0}};
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
                return LaneInfo{analysis.getStride(*arg0), analysis.isAligned(*arg0)};
        }
        if((intrinsic->opCode == "sdiv" || intrinsic->opCode == "udiv") && arg0 && arg1 &&
            analysis.isAligned(*arg0) && getMultipleOf16Shift(*arg1))
            // (16 * u + i) / 2^k is u / 2^(k - 4) in all lanes (u is not negative)
            return LaneInfo{int64_t{0}};
        return LaneInfo{};
    }
    return LaneInfo{};
}

/*
 * Determines all values which differ between the work-items of a QPU.
 */
static Analysis analyzeVaryingValues(Method& method, const ControlFlow& cf, uint8_t vectorWidth)
{
    Analysis analysis;
    auto localBlocks = determineLocalBlocks(method);
    bool changed = true;
    while(changed)
    {
        changed = false;
        for(auto& block : method)
        {
            // whether the flags were last set from a varying value
            bool varyingFlags = false;
            // whether the replication register (r5) was last written with a varying value, e.g. for a splat
            bool varyingReplication = false;
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
                        // the chunks of work-items start at multiples of 16 in x, so the local ID in lane i is 16 * u + i
                        if(dim && *dim == 0 && out)
                            markVarying(analysis, out,
                                LaneInfo{int64_t{1}, vectorWidth == 1 && call->methodName == intrinsics::FUNCTION_NAME_LOCAL_ID},
                                changed);
                        // dimensions 1 and 2 are uniform, since SIMT mode is only used for 1-dimensional work-groups
                        continue;
                    }
                }

                bool readsVarying = false;
                instr->forReadLocals([&](const Local* loc, const IntermediateInstruction&) {
                    if(analysis.varying.find(loc) != analysis.varying.end())
                        readsVarying = true;
                });
                if(varyingReplication && readsReplicationRegister(*instr))
                    readsVarying = true;
                if(writesReplicationRegister(*instr))
                    varyingReplication = readsVarying;
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

        // values written in divergent regions (for some lanes only) and used outside of their block differ between
        // the lanes, e.g. the value of a phi-node after a divergent if-else
        std::vector<Region> regions;
        findDivergentRegions(cf, [&](const Value& cond) { return analysis.isVarying(cond); }, regions);
        for(const auto& region : regions)
        {
            for(auto b : region.blocks)
            {
                for(const auto& instr : *cf.blocks[b])
                {
                    auto out = instr ? instr->checkOutputLocal() : nullptr;
                    if(out && !out->type.isLabelType() && !isMemoryWrite(*instr) &&
                        isUsedOutside(localBlocks, out, *cf.blocks[b]))
                        markVarying(analysis, out, LaneInfo{}, changed);
                }
            }
        }
    }
    return analysis;
}

/*
 * Checks whether the kernel can be run in SIMT mode with the given varying values. Returns the reason why not, or an
 * empty string.
 */
/*
 * In vector kernels, rotations are only supported where the front-end uses them to access single vector elements: they
 * are only used by replications (splats), element insertions and the extraction of the first element. Then the lanes
 * read from the rotated value never cross the boundary between work-items (see convertVectorOperations).
 */
static bool isElementAccessRotation(const MoveOperation& rotation)
{
    auto out = rotation.checkOutputLocal();
    if(!out)
        return false;
    bool supported = true;
    out->forUsers(LocalUse::Type::READER, [&](const LocalUser* user) {
        auto move = dynamic_cast<const MoveOperation*>(user);
        if(!move || move->getVectorRotation())
            supported = false;
        else if(move->hasConditionalExecution() || move->getOutput()->hasRegister(REG_REPLICATE_ALL))
            // element insertion or replication
            return;
        else if(!move->getOutput()->type.isScalarType())
            supported = false;
    });
    return supported;
}

/*
 * The per-lane addresses of work-item dependent pointers as vectors of 16 addresses. The pointers themselves stay
 * scalars in SIMT mode (holding the per-lane addresses), but the code generation may treat scalars as the same in all
 * lanes, e.g. the instruction combining or the register allocation fix-ups grouping scalars into the lanes of a
 * register and replicating them again. So every pointer accessed lane by lane gets a "shadow" vector, written next to
 * (and under the same condition as) every write of the pointer. Supported are pointers written by adding or
 * subtracting offsets and by moves.
 */
static bool isAddressVectorSupported(
    const Analysis& analysis, const Value& address, FastSet<const Local*>& visited, unsigned depth = 0)
{
    auto loc = address.checkLocal();
    if(!analysis.isVarying(address) || !address.type.getPointerType())
        return true;
    if(!visited.emplace(loc).second)
        // already checked or being checked (pointers written in loops)
        return true;
    if(depth > 32)
        return false;
    for(const auto& user : loc->getUsers())
    {
        if(!user.second.writesLocal())
            continue;
        auto writer = user.first;
        if(dynamic_cast<const MemoryInstruction*>(writer))
            // writes the memory the pointer points to, not the pointer
            continue;
        if(writer->hasUnpackMode() || writer->hasPackMode() || writer->doesSetFlag())
            return false;
        if(auto move = dynamic_cast<const MoveOperation*>(writer))
        {
            if(move->getVectorRotation() || !isAddressVectorSupported(analysis, move->getSource(), visited, depth + 1))
                return false;
            continue;
        }
        auto op = dynamic_cast<const Operation*>(writer);
        if(!op || (op->op != OP_ADD && op->op != OP_SUB) || !op->getArgument(1) ||
            !isAddressVectorSupported(analysis, op->assertArgument(0), visited, depth + 1) ||
            !isAddressVectorSupported(analysis, op->assertArgument(1), visited, depth + 1))
            return false;
    }
    return true;
}

class AddressVectors
{
public:
    explicit AddressVectors(Method& method, const Analysis& analysis) : method(method), analysis(analysis) {}

    // The vector of addresses of the pointer, inserting its calculation next to the writes of the pointer
    Value get(const Value& address)
    {
        auto loc = address.checkLocal();
        if(!analysis.isVarying(address) || !address.type.getPointerType())
            // the same in all lanes, or an offset (converted to a vector by the conversion)
            return address;
        auto shadowIt = shadows.find(loc);
        if(shadowIt != shadows.end())
            return shadowIt->second;
        auto shadow = method.addNewLocal(TYPE_INT32.toVectorType(SIMT_WIDTH), "%simt_lane_addresses");
        shadows.emplace(loc, shadow);
        if(walkers.empty())
        {
            for(auto& block : method)
            {
                for(auto it = block.walk(); !it.isEndOfBlock(); it.nextInBlock())
                {
                    if(it.has())
                        walkers.emplace(it.get(), it);
                }
            }
        }
        std::vector<const IntermediateInstruction*> writers;
        for(const auto& user : loc->getUsers())
        {
            if(user.second.writesLocal() && !dynamic_cast<const MemoryInstruction*>(user.first))
                writers.push_back(user.first);
        }
        for(auto writer : writers)
        {
            auto writerIt = walkers.at(writer);
            std::unique_ptr<ExtendedInstruction> calculation;
            if(auto move = dynamic_cast<const MoveOperation*>(writer))
                calculation = std::make_unique<MoveOperation>(shadow, get(move->getSource()));
            else
            {
                auto op = dynamic_cast<const Operation*>(writer);
                calculation =
                    std::make_unique<Operation>(op->op, shadow, get(op->assertArgument(0)), get(op->assertArgument(1)));
            }
            calculation->setCondition(dynamic_cast<const ExtendedInstruction*>(writer)->getCondition());
            auto insertIt = writerIt.copy().nextInBlock();
            insertIt.emplace(std::move(calculation));
            walkers.emplace(insertIt.get(), insertIt);
        }
        return shadow;
    }

private:
    Method& method;
    const Analysis& analysis;
    FastMap<const Local*, Value> shadows;
    FastMap<const IntermediateInstruction*, InstructionWalker> walkers;
};

/*
 * Whether the work-items access their 32-bit elements at addresses which are not consecutive across the lanes (e.g.
 * transposed or strided accesses, or computed indices). Such accesses are run lane by lane, see insertScatteredAccess.
 */
static bool isScatteredAccess(const Analysis& analysis, const MemoryInstruction& mem, uint8_t vectorWidth)
{
    if(vectorWidth > 1 || (mem.op != MemoryOperation::READ && mem.op != MemoryOperation::WRITE))
        return false;
    bool isRead = mem.op == MemoryOperation::READ;
    const Value& address = isRead ? mem.getSource() : mem.getDestination();
    if(!analysis.isVarying(address))
        return false;
    auto elementType = isRead ? mem.getSourceElementType() : mem.getDestinationElementType();
    if(!elementType.isScalarType() || elementType.getScalarBitCount() != 32)
        return false;
    auto stride = analysis.getStride(address);
    FastSet<const Local*> visited;
    return (!stride || *stride != 4) && isAddressVectorSupported(analysis, address, visited);
}

static std::string checkVaryingValues(const Method& method, const Analysis& analysis, const ControlFlow& cf,
    const std::vector<Region>& regions, uint8_t vectorWidth)
{
    if(vectorWidth > 1 && !regions.empty())
        return "divergent control flow in a vector kernel";
    // the blocks of the divergent regions (including the branch blocks), and the region blocks only
    FastSet<const BasicBlock*> divergentBlocks;
    FastSet<const BasicBlock*> maskedBlocks;
    for(const auto& region : regions)
    {
        divergentBlocks.emplace(cf.blocks[region.entryBlock]);
        for(auto b : region.blocks)
        {
            divergentBlocks.emplace(cf.blocks[b]);
            maskedBlocks.emplace(cf.blocks[b]);
        }
    }

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
                if(!branch->isUnconditional() && varyingFlags && divergentBlocks.find(&block) == divergentBlocks.end())
                    return "unsupported divergent branch: " + branch->to_string();
            }
            auto call = dynamic_cast<const MethodCall*>(instr.get());
            if(call && isBarrier(*call) && maskedBlocks.find(&block) != maskedBlocks.end())
                // OpenCL requires all work-items to reach the barrier, i.e. the kernel is invalid or the condition was
                // wrongly determined to differ between the work-items
                return "barrier in divergent control flow: " + call->to_string();
            if(auto mem = dynamic_cast<const MemoryInstruction*>(instr.get()))
            {
                const auto& address = mem->op == MemoryOperation::READ ? mem->getSource() : mem->getDestination();
                auto elementType =
                    mem->op == MemoryOperation::READ ? mem->getSourceElementType() : mem->getDestinationElementType();
                if(analysis.isVarying(address))
                {
                    // the work-items access consecutive elements (scalars or vectors). In vector kernels, the work-items
                    // may also read one element of consecutive vectors (a vector load split up by the front-end).
                    auto stride = analysis.getStride(address);
                    auto elementBytes =
                        static_cast<int64_t>(elementType.getScalarBitCount() / 8 * elementType.getVectorWidth());
                    bool isVectorElementRead = vectorWidth > 1 && mem->op == MemoryOperation::READ &&
                        elementType.isScalarType() && stride && *stride == elementBytes * vectorWidth;
                    if(!stride || (*stride != elementBytes && !isVectorElementRead) ||
                        !isSupportedType(elementType, vectorWidth))
                    {
                        if(!isScatteredAccess(analysis, *mem, vectorWidth))
                            return "memory access which is not contiguous across work-items: " + mem->to_string();
                    }
                    else if(mem->op == MemoryOperation::WRITE && maskedBlocks.find(&block) != maskedBlocks.end() &&
                        elementType.getScalarBitCount() != 32)
                        // see insertMaskedStore
                        return "store of 8- or 16-bit values in divergent code: " + mem->to_string();
                }
                else if(mem->op == MemoryOperation::WRITE && analysis.isVarying(mem->getSource()))
                    return "work-items writing different values to the same address: " + mem->to_string();
                else if(mem->op == MemoryOperation::WRITE && vectorWidth > 1 && elementType.isVectorType())
                    // the store would be taken for a store of the work-items' elements, see lowerWriteRAM
                    return "vector store to a work-group uniform address: " + mem->to_string();
            }
            if(readsVarying && !dynamic_cast<const Operation*>(instr.get()) &&
                !dynamic_cast<const MoveOperation*>(instr.get()) &&
                !dynamic_cast<const IntrinsicOperation*>(instr.get()) &&
                !dynamic_cast<const MemoryInstruction*>(instr.get()) && !dynamic_cast<const MethodCall*>(instr.get()) &&
                !dynamic_cast<const Branch*>(instr.get()))
                return "unsupported instruction reading a work-item dependent value: " + instr->to_string();
            if(auto move = dynamic_cast<const MoveOperation*>(instr.get()))
            {
                bool isVectorAccess = vectorWidth > 1 && (move->getSource().type.isVectorType() || readsVarying);
                if(move->getVectorRotation() && isVectorAccess && !isElementAccessRotation(*move))
                    return "unsupported vector shuffle: " + move->to_string();
                if(readsVarying && move->getVectorRotation() && vectorWidth == 1)
                    return "vector rotation of a work-item dependent value: " + move->to_string();
            }

            if(instr->doesSetFlag())
                varyingFlags = readsVarying;
        }
    }
    return "";
}

/*
 * Vector kernels (see determineVectorWidth): every work-item uses vectorWidth adjacent SIMD lanes.
 */

static Value rotate(Method& method, InstructionWalker& it, const Value& src, uint32_t offset, Direction direction)
{
    // copy into a 16-element value first, so the rotation is not shortened to the source's vector width
    auto source = assign(it, src.type.toVectorType(SIMT_WIDTH), "%simt_rotation_source") = src;
    auto result = method.addNewLocal(src.type.toVectorType(SIMT_WIDTH), "%simt_rotated");
    it = insertVectorRotation(it, source, Value(Literal(offset), TYPE_INT8), result, direction);
    return result;
}

/*
 * dest[lane] = src[first lane of the work-item]: replicates the first element of every work-item to all of its lanes
 * (the vector kernel's version of replicating element 0 to all lanes).
 */
static void insertWorkItemReplication(Method& method, InstructionWalker& it, const Value& src, const Value& dest,
    uint8_t vectorWidth, const Value& elementIndex)
{
    assign(it, dest) = src;
    for(uint32_t offset = 1; offset < vectorWidth; ++offset)
    {
        auto rotated = rotate(method, it, src, offset, Direction::UP);
        assign(it, NOP_REGISTER) = (elementIndex ^ Value(Literal(offset), TYPE_INT8), SetFlag::SET_FLAGS);
        assign(it, dest) = (rotated, COND_ZERO_SET);
    }
}

/*
 * dest[lane] = src[lane / vectorWidth]: distributes the values of consecutive elements to the work-items, e.g. the
 * values of a scalar load of 16 consecutive elements (of which the first 16 / vectorWidth belong to the work-items).
 */
static void insertExpansion(Method& method, InstructionWalker& it, const Value& src, const Value& dest,
    uint8_t vectorWidth, const Value& elementIndex)
{
    auto spread = assign(it, src.type.toVectorType(SIMT_WIDTH), "%simt_spread") = src;
    // first move the value of work-item w into its first lane w * vectorWidth
    for(uint32_t w = 1; w < SIMT_WIDTH / vectorWidth; ++w)
    {
        auto rotated = rotate(method, it, src, w * (vectorWidth - 1u), Direction::UP);
        assign(it, NOP_REGISTER) =
            (ELEMENT_NUMBER_REGISTER ^ Value(Literal(w * vectorWidth), TYPE_INT8), SetFlag::SET_FLAGS);
        assign(it, spread) = (rotated, COND_ZERO_SET);
    }
    insertWorkItemReplication(method, it, spread, dest, vectorWidth, elementIndex);
}

/*
 * dest[w] = src[w * vectorWidth]: collects the (scalar) values of the work-items into consecutive elements, e.g. for a
 * scalar store to consecutive addresses.
 */
static void insertCompaction(Method& method, InstructionWalker& it, const Value& src, const Value& dest,
    uint8_t vectorWidth)
{
    assign(it, dest) = src;
    for(uint32_t w = 1; w < SIMT_WIDTH / vectorWidth; ++w)
    {
        auto rotated = rotate(method, it, src, w * (vectorWidth - 1u), Direction::DOWN);
        assign(it, NOP_REGISTER) = (ELEMENT_NUMBER_REGISTER ^ Value(Literal(w), TYPE_INT8), SetFlag::SET_FLAGS);
        assign(it, dest) = (rotated, COND_ZERO_SET);
    }
}

/*
 * Converts the method: varying values become vectors with one element per work-item.
 */
static void convertToSIMT(Method& method, const Analysis& analysis, uint8_t vectorWidth,
    FastMap<const IntermediateInstruction*, Value>& scatteredAccesses)
{
    AddressVectors addressVectors(method, analysis);
    const auto workItemsPerQPU = static_cast<uint8_t>(SIMT_WIDTH / vectorWidth);
    auto startIt = method.walkAllInstructions();
    if(!startIt.isEndOfMethod() && startIt.get<BranchLabel>())
        startIt.nextInBlock();

    // 1. new 16-element locals for all varying non-pointer values (in vector kernels, a scalar value is repeated in
    // all lanes of its work-item) and in vector kernels for all vectors. Pointers keep their type: they hold the
    // per-lane address, but only the address in lane 0 is used by the (converted) memory accesses below.
    FastMap<const Local*, Value> replacements;
    FastSet<const Local*> vectorLocals;
    auto addReplacement = [&](const Local* loc) {
        if(loc->type.getPointerType() || loc->type.isLabelType() || replacements.find(loc) != replacements.end())
            return;
        replacements.emplace(loc, method.addNewLocal(loc->type.toVectorType(SIMT_WIDTH), loc->name, "simt"));
    };
    for(const auto& entry : analysis.varying)
        addReplacement(entry.first);

    // vector kernels: lane = work-item * vectorWidth + element
    Value elementIndex = ELEMENT_NUMBER_REGISTER;
    Value workItemIndex = ELEMENT_NUMBER_REGISTER;
    if(vectorWidth > 1)
    {
        elementIndex = method.addNewLocal(ELEMENT_NUMBER_REGISTER.type, "%simt_element_index");
        workItemIndex = method.addNewLocal(ELEMENT_NUMBER_REGISTER.type, "%simt_work_item_index");
        for(auto& block : method)
        {
            for(auto& instr : block)
            {
                if(!instr)
                    continue;
                instr->forUsedLocals([&](const Local* loc, LocalUse::Type, const IntermediateInstruction&) {
                    if(!loc->type.getPointerType() && !loc->type.isLabelType() && loc->type.isVectorType())
                    {
                        vectorLocals.emplace(loc);
                        addReplacement(loc);
                    }
                });
                // the front-end uses the element number for element accesses, which refer to the element of the
                // work-item's vector
                const auto args = instr->getArguments();
                for(std::size_t i = 0; i < args.size(); ++i)
                {
                    if(args[i].hasRegister(REG_ELEMENT_NUMBER))
                        instr->setArgument(i, elementIndex);
                }
            }
        }
        // inserted after the loop above, so the element number read here is not replaced
        uint32_t shift = vectorWidth == 2 ? 1 : (vectorWidth == 4 ? 2 : 3);
        assign(startIt, elementIndex) =
            (ELEMENT_NUMBER_REGISTER & Value(Literal(vectorWidth - 1u), TYPE_INT8));
        assign(startIt, workItemIndex) = as_unsigned{ELEMENT_NUMBER_REGISTER} >> Value(Literal(shift), TYPE_INT8);
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
            // run-time passes the first local ID of every QPU) plus the index of the lane's work-item
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
                        OP_ADD, Value(replacement->second), Value(base), Value(workItemIndex)));
                    CPPLOG_LAZY(logging::Level::DEBUG,
                        log << "SIMT: work-item ID per lane: " << next->to_string() << logging::endl);
                    it.nextInBlock().nextInBlock();
                    continue;
                }
            }

            // 3. memory accesses with per-work-item contiguous addresses: access 16 elements at the address of lane 0
            if(auto mem = it.get<MemoryInstruction>())
            {
                if(isScatteredAccess(analysis, *mem, vectorWidth))
                {
                    // the per-lane addresses are calculated as a vector here, the value is converted to a vector below.
                    // The access is split up into the lanes after the linearization of the control flow, which knows
                    // the lanes to access.
                    auto address = mem->op == MemoryOperation::READ ? mem->getSource() : mem->getDestination();
                    scatteredAccesses.emplace(mem, addressVectors.get(address));
                    CPPLOG_LAZY(logging::Level::DEBUG,
                        log << "SIMT: memory access per lane: " << mem->to_string() << logging::endl);
                    it.nextInBlock();
                    continue;
                }
                bool isRead = mem->op == MemoryOperation::READ;
                const Value address = isRead ? mem->getSource() : mem->getDestination();
                auto elementType = isRead ? mem->getSourceElementType() : mem->getDestinationElementType();
                auto ptrType = address.type.getPointerType();
                if(analysis.isVarying(address))
                {
                    // scalars in vector kernels: the work-items' elements are in the first lanes in memory
                    bool isScalarOfVectorKernel = vectorWidth > 1 && elementType.isScalarType();
                    auto accessType = elementType.toVectorType(
                        isScalarOfVectorKernel && !isRead ? workItemsPerQPU : static_cast<uint8_t>(SIMT_WIDTH));
                    auto vectorPtr = method.addNewLocal(
                        method.createPointerType(accessType, ptrType->addressSpace, ptrType->alignment),
                        address.checkLocal()->name, "simt_ptr");
                    it.emplace(std::make_unique<MoveOperation>(vectorPtr, address));
                    it.nextInBlock();
                    mem = it.get<MemoryInstruction>();
                    // the address is the source of a read and the output of a write
                    if(isRead)
                    {
                        mem->setArgument(0, vectorPtr);
                        if(isScalarOfVectorKernel)
                        {
                            const Value dest = mem->getDestination();
                            auto loaded = method.addNewLocal(accessType, "%simt_loaded");
                            mem->setOutput(loaded);
                            it.nextInBlock();
                            auto stride = analysis.getStride(address);
                            if(stride && *stride == static_cast<int64_t>(elementType.getScalarBitCount() / 8))
                                // consecutive scalars: the value of work-item w is in lane w
                                insertExpansion(method, it, loaded, dest, vectorWidth, elementIndex);
                            else
                                // the same element of consecutive vectors: the value of work-item w is in its first
                                // lane w * vectorWidth already
                                insertWorkItemReplication(method, it, loaded, dest, vectorWidth, elementIndex);
                            continue;
                        }
                    }
                    else
                    {
                        mem->setOutput(vectorPtr);
                        const Value source = mem->getSource();
                        auto sourceLocal = source.checkLocal();
                        if(sourceLocal ? !analysis.isVarying(source) && !vectorLocals.count(sourceLocal) :
                                         source.isAllSame())
                        {
                            // a work-group uniform value (e.g. out[id] = sum) is the same for all work-items, but
                            // might only be set in the first lane. (The varying values are converted to vectors
                            // later.)
                            auto replicated =
                                method.addNewLocal(source.type.toVectorType(SIMT_WIDTH), "%simt_uniform_value");
                            if(source.isAllSame())
                                assign(it, replicated) = source;
                            else
                                it = insertReplication(it, source, replicated);
                            mem = it.get<MemoryInstruction>();
                            mem->setArgument(0, replicated);
                        }
                        if(isScalarOfVectorKernel)
                        {
                            auto compacted = method.addNewLocal(accessType, "%simt_compacted");
                            insertCompaction(method, it, mem->getSource(), compacted, vectorWidth);
                            mem->setArgument(0, compacted);
                        }
                    }
                    CPPLOG_LAZY(logging::Level::DEBUG,
                        log << "SIMT: memory access per work-item: " << mem->to_string() << logging::endl);
                }
                else if(vectorWidth == 1 && isRead && !elementType.isVectorType())
                {
                    // A value loaded from a work-group uniform address is the same for all work-items, but a load via
                    // the VPM only sets it in the first lane
                    const Value dest = mem->getDestination();
                    auto loaded = method.addNewLocal(dest.type, "%simt_loaded");
                    mem->setOutput(loaded);
                    it.nextInBlock();
                    it = insertReplication(it, loaded, dest);
                    continue;
                }
                else if(vectorWidth > 1 && isRead && elementType.isVectorType())
                {
                    // a vector at a uniform address is the same for all work-items
                    const Value dest = mem->getDestination();
                    auto loaded = method.addNewLocal(elementType, "%simt_loaded");
                    mem->setOutput(loaded);
                    it.nextInBlock();
                    auto repeated = method.addNewLocal(elementType.toVectorType(SIMT_WIDTH), "%simt_repeated");
                    it = insertVectorReplication(it, method, loaded, repeated);
                    assign(it, dest) = repeated;
                    continue;
                }
            }

            // 4. vector kernels: replicating element 0 (e.g. for splats and element extraction) replicates the first
            // element of every work-item, and so does taking element 0 of a vector
            auto move = it.get<MoveOperation>();
            auto source = move ? move->getSource().checkLocal() : nullptr;
            // also an element of a vector moved to the first lane by a rotation
            auto sourceWriter = source ? dynamic_cast<const MoveOperation*>(source->getSingleWriter()) : nullptr;
            bool isPerWorkItem = source &&
                (vectorLocals.find(source) != vectorLocals.end() || analysis.isVarying(move->getSource()) ||
                    (sourceWriter && sourceWriter->getVectorRotation()));
            if(vectorWidth > 1 && isPerWorkItem && !move->getVectorRotation() && !move->hasConditionalExecution())
            {
                if(move->getOutput()->hasRegister(REG_REPLICATE_ALL))
                {
                    auto replicated = method.addNewLocal(source->type.toVectorType(SIMT_WIDTH), "%simt_replicated");
                    insertWorkItemReplication(method, it, move->getSource(), replicated, vectorWidth, elementIndex);
                    // the following read of the replicated value reads ours instead
                    for(auto reader = it.copy().nextInBlock(); !reader.isEndOfBlock(); reader.nextInBlock())
                    {
                        if(!reader.has())
                            continue;
                        const auto args = reader->getArguments();
                        bool found = false;
                        for(std::size_t i = 0; i < args.size(); ++i)
                        {
                            if(args[i].hasRegister(REG_REPLICATE_ALL) || args[i].hasRegister(REG_ACC5))
                            {
                                reader->setArgument(i, replicated);
                                found = true;
                            }
                        }
                        if(found)
                            break;
                    }
                    it.erase();
                    continue;
                }
                auto out = move->checkOutputLocal();
                if(out && vectorLocals.find(source) != vectorLocals.end() &&
                    vectorLocals.find(out) == vectorLocals.end())
                {
                    // vector to scalar: element 0
                    auto replicated = method.addNewLocal(source->type.toVectorType(SIMT_WIDTH), "%simt_element");
                    insertWorkItemReplication(method, it, move->getSource(), replicated, vectorWidth, elementIndex);
                    move->setSource(Value(replicated));
                }
            }
            it.nextInBlock();
        }
    }

    // 5. redirect all uses of the varying values to their vector versions
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

    // 6. memory stores only write back the elements of work-items in the work-group, which needs the local size and
    // the QPU's first local ID (see lowerWriteRAM in periphery/VPM.cpp). Read them here, so the UNIFORMs are loaded at
    // the start of the kernel.
    for(auto type : {BuiltinLocal::Type::LOCAL_SIZES, BuiltinLocal::Type::LOCAL_IDS})
    {
        auto builtin = method.findOrCreateBuiltin(type);
        startIt.emplace(std::make_unique<MoveOperation>(
            method.addNewLocal(TYPE_INT32, "%simt" + builtin->name.substr(1)), builtin->createReference()));
        startIt.nextInBlock();
    }

    method.metaData.mergedWorkItemsFactor = workItemsPerQPU;
}

static DataType getMaskType()
{
    return TYPE_BOOL.toVectorType(SIMT_WIDTH);
}

static const Local* createLabel(Method& method, const std::string& name)
{
    return method.addNewLocal(TYPE_LABEL, "", name).local();
}

// Inserts a conditional branch before the given position, the walker keeps pointing to the same instruction
static void insertBranch(InstructionWalker& it, const Local* target, BranchCond cond)
{
    it.emplace(std::make_unique<Branch>(target, cond));
    it.nextInBlock();
}

/*
 * The lanes running work-items of the work-group: the first min(16, local_size(0) - first local ID of the QPU) lanes.
 * Calculated at the start of the kernel.
 */
static Value insertEntryMask(Method& method)
{
    auto it = method.walkAllInstructions();
    if(it.get<BranchLabel>())
        it.nextInBlock();
    auto localSizes = method.findOrCreateBuiltin(BuiltinLocal::Type::LOCAL_SIZES)->createReference();
    auto localIds = method.findOrCreateBuiltin(BuiltinLocal::Type::LOCAL_IDS)->createReference();
    auto localSizeX = assign(it, TYPE_INT32, "%simt_local_size_x") = (localSizes & Value(Literal(0xFFu), TYPE_INT32));
    auto firstLocalId = assign(it, TYPE_INT32, "%simt_first_local_id") = (localIds & Value(Literal(0xFFu), TYPE_INT32));
    auto remainingItems = assign(it, TYPE_INT32, "%simt_remaining_items") = (localSizeX - firstLocalId);
    auto cond = assignNop(it) = (as_signed{ELEMENT_NUMBER_REGISTER} < as_signed{remainingItems});
    auto mask = method.addNewLocal(getMaskType(), "%simt_entry_mask");
    assign(it, mask) = (BOOL_TRUE, cond);
    assign(it, mask) = (BOOL_FALSE, cond.invert());
    return mask;
}

// The mask receiving the lanes taking an edge, if any
using EdgeMaskResolver = std::function<const Value*(const Edge& edge)>;

/*
 * Adds the lanes of the given block taking the edges of the block exit to the masks of the target blocks. Edges to
 * blocks without a mask (the merge block of the region) are ignored.
 */
static void insertEdgeMasks(Method& method, InstructionWalker it, const BlockExit& exit, const Value& blockMask,
    const EdgeMaskResolver& resolveMask, const FastMap<const Local*, const Local*>& renamed)
{
    Optional<Value> condition;
    for(const auto& edge : exit.edges)
    {
        auto targetMaskPtr = resolveMask(edge);
        if(!targetMaskPtr)
            continue;
        const auto& targetMask = *targetMaskPtr;
        if(edge.kind == EdgeKind::ALWAYS)
        {
            assign(it, targetMask) = (targetMask | blockMask);
            continue;
        }
        if(!condition)
        {
            auto cond = *exit.condition;
            if(auto loc = cond.checkLocal())
            {
                auto renamedIt = renamed.find(loc);
                if(renamedIt != renamed.end())
                    cond = renamedIt->second->createReference();
            }
            if(!cond.type.isVectorType())
            {
                // a uniform condition (inside a divergent region), make sure it is set in all lanes
                auto vector = method.addNewLocal(cond.type.toVectorType(SIMT_WIDTH), "%simt_condition");
                it = insertReplication(it, cond, vector);
                cond = vector;
            }
            condition = cond;
        }
        assign(it, NOP_REGISTER) = (*condition, SetFlag::SET_FLAGS);
        assign(it, targetMask) =
            (targetMask | blockMask, edge.kind == EdgeKind::ON_TRUE ? COND_ZERO_CLEAR : COND_ZERO_SET);
    }
}

/*
 * Replaces the store (of 16 elements to contiguous addresses) by a store which only writes the elements of the lanes
 * in the block mask:
 *
 *   if all lanes of the work-group are active: the store as before
 *   else: for(lane = 0; lane < 16; ++lane) if(mask[lane]) store value[lane] to address + lane * 4
 *
 * A VPM row write can't be masked, but a DMA of a single word writes exactly that word.
 */
/*
 * Replaces the access of the lanes' elements at their own (not consecutive) addresses by a loop over the lanes in the
 * mask, each accessing its single element:
 *
 *   for(lane = 0; lane < 16; ++lane) if(mask[lane]) value[lane] = *address[lane] (or *address[lane] = value[lane])
 *
 * The lanes not in the mask may have any address, which must not be accessed (the QPUs have no MMU).
 */
static void insertScatteredAccess(Method& method, InstructionWalker it, const Value& mask, const Value& laneAddresses)
{
    auto mem = it.get<MemoryInstruction>();
    const bool isRead = mem->op == MemoryOperation::READ;
    const Value address = isRead ? mem->getSource() : mem->getDestination();
    const Value value = isRead ? mem->getDestination() : mem->getSource();
    auto elementType = (isRead ? mem->getSourceElementType() : mem->getDestinationElementType()).getElementType();
    auto ptrType = address.type.getPointerType();
    auto laneAddressType = method.createPointerType(elementType, ptrType->addressSpace, ptrType->alignment);

    auto checkLabel = createLabel(method, "%simt_lane_check");
    auto accessLabel = createLabel(method, "%simt_lane_access");
    auto nextLabel = createLabel(method, "%simt_next_lane");
    auto doneLabel = createLabel(method, "%simt_lanes_done");

    auto lane = method.addNewLocal(TYPE_INT32, "%simt_lane");
    assign(it, lane) = INT_ZERO;

    // is the lane in the mask?
    auto checkIt = method.emplaceLabel(it, std::make_unique<BranchLabel>(*checkLabel));
    checkIt.nextInBlock();
    auto laneActive = method.addNewLocal(TYPE_BOOL, "%simt_lane_active");
    checkIt = insertVectorExtraction(checkIt, method, mask, lane, laneActive);
    BranchCond activeCond = BRANCH_ALWAYS;
    std::tie(checkIt, activeCond) = insertBranchCondition(method, checkIt, laneActive);
    insertBranch(checkIt, accessLabel, activeCond);
    insertBranch(checkIt, nextLabel, activeCond.invert());

    // the access of the lane's element
    auto accessIt = method.emplaceLabel(checkIt, std::make_unique<BranchLabel>(*accessLabel));
    accessIt.nextInBlock();
    auto laneAddress = method.addNewLocal(laneAddressType, "%simt_lane_address");
    if(auto base = address.checkLocal() ? address.local()->getBase(true) : nullptr)
        // the memory area accessed, e.g. for the selection of the memory access type
        laneAddress.local()->set(ReferenceData(*base, ANY_ELEMENT));
    accessIt = insertVectorExtraction(accessIt, method, laneAddresses, lane, laneAddress);
    auto laneValue = method.addNewLocal(elementType, "%simt_lane_value");
    if(isRead)
    {
        accessIt.emplace(
            std::make_unique<MemoryInstruction>(MemoryOperation::READ, Value(laneValue), Value(laneAddress)));
        accessIt.nextInBlock();
        accessIt = insertVectorInsertion(accessIt, method, value, lane, laneValue);
    }
    else
    {
        if(value.type.isVectorType())
            accessIt = insertVectorExtraction(accessIt, method, value, lane, laneValue);
        else
            // a work-group uniform value
            assign(accessIt, laneValue) = value;
        accessIt.emplace(
            std::make_unique<MemoryInstruction>(MemoryOperation::WRITE, Value(laneAddress), Value(laneValue)));
        accessIt.nextInBlock();
    }
    // the original access
    accessIt.erase();

    // the next lane
    auto nextIt = method.emplaceLabel(accessIt, std::make_unique<BranchLabel>(*nextLabel));
    nextIt.nextInBlock();
    assign(nextIt, lane) = (lane + INT_ONE);
    assign(nextIt, NOP_REGISTER) = (lane - Value(Literal(static_cast<uint32_t>(SIMT_WIDTH)), TYPE_INT32),
        SetFlag::SET_FLAGS);
    // the lane counter is the same in all SIMD elements
    insertBranch(nextIt, checkLabel, BRANCH_ALL_N_SET);
    insertBranch(nextIt, doneLabel, BRANCH_ANY_N_CLEAR);

    // the remaining instructions of the original block
    auto doneIt = method.emplaceLabel(nextIt, std::make_unique<BranchLabel>(*doneLabel));
    static_cast<void>(doneIt);
}

static void insertMaskedStore(Method& method, InstructionWalker it, const Value& blockMask, const Value& entryMask)
{
    auto store = it.get<MemoryInstruction>();
    const Value value = store->getSource();
    const Value address = store->getDestination();
    auto elementType = store->getDestinationElementType().getElementType();
    auto ptrType = address.type.getPointerType();
    auto laneAddressType = method.createPointerType(elementType, ptrType->addressSpace, ptrType->alignment);

    auto allLabel = createLabel(method, "%simt_store_all");
    auto laneLoopLabel = createLabel(method, "%simt_store_lanes");
    auto laneStoreLabel = createLabel(method, "%simt_store_lane");
    auto laneNextLabel = createLabel(method, "%simt_store_next_lane");
    auto doneLabel = createLabel(method, "%simt_store_done");

    // all lanes of the work-group active?
    auto inactive = assign(it, getMaskType(), "%simt_inactive_lanes") = (blockMask ^ entryMask);
    assign(it, NOP_REGISTER) = (inactive, SetFlag::SET_FLAGS);
    insertBranch(it, allLabel, BRANCH_ALL_Z_SET);
    insertBranch(it, laneLoopLabel, BRANCH_ANY_Z_CLEAR);

    // block with the original store
    auto allIt = method.emplaceLabel(it, std::make_unique<BranchLabel>(*allLabel));
    auto allBlock = allIt.getBasicBlock();
    auto afterIt = allIt.nextInBlock().nextInBlock();

    // the loop over the lanes, starting before the instructions after the store
    auto loopIt = method.emplaceLabel(afterIt, std::make_unique<BranchLabel>(*laneLoopLabel));
    allBlock->walkEnd().emplace(std::make_unique<Branch>(doneLabel));
    loopIt.nextInBlock();
    auto lane = method.addNewLocal(TYPE_INT32, "%simt_lane");
    assign(loopIt, lane) = INT_ZERO;

    auto checkIt = method.emplaceLabel(loopIt, std::make_unique<BranchLabel>(*createLabel(method, "%simt_lane_check")));
    checkIt.nextInBlock();
    auto laneActive = method.addNewLocal(TYPE_BOOL, "%simt_lane_active");
    checkIt = insertVectorExtraction(checkIt, method, blockMask, lane, laneActive);
    BranchCond activeCond = BRANCH_ALWAYS;
    std::tie(checkIt, activeCond) = insertBranchCondition(method, checkIt, laneActive);
    insertBranch(checkIt, laneStoreLabel, activeCond);
    insertBranch(checkIt, laneNextLabel, activeCond.invert());

    auto storeIt = method.emplaceLabel(checkIt, std::make_unique<BranchLabel>(*laneStoreLabel));
    storeIt.nextInBlock();
    auto laneValue = method.addNewLocal(elementType, "%simt_lane_value");
    storeIt = insertVectorExtraction(storeIt, method, value, lane, laneValue);
    auto laneOffset = assign(storeIt, TYPE_INT32, "%simt_lane_offset") =
        (lane << Value(Literal(2u), TYPE_INT8)); // 32-bit elements only, see checkVaryingValues
    auto laneAddress = method.addNewLocal(laneAddressType, "%simt_lane_address");
    assign(storeIt, laneAddress) = (address + laneOffset);
    storeIt.emplace(std::make_unique<MemoryInstruction>(MemoryOperation::WRITE, Value(laneAddress), Value(laneValue)));
    storeIt.nextInBlock();

    auto nextIt = method.emplaceLabel(storeIt, std::make_unique<BranchLabel>(*laneNextLabel));
    nextIt.nextInBlock();
    assign(nextIt, lane) = (lane + INT_ONE);
    assign(nextIt, NOP_REGISTER) = (lane - Value(Literal(static_cast<uint32_t>(SIMT_WIDTH)), TYPE_INT32),
        SetFlag::SET_FLAGS);
    // the lane counter is the same in all SIMD elements
    insertBranch(nextIt, checkIt.getBasicBlock()->getLabel()->getLabel(), BRANCH_ALL_N_SET);
    insertBranch(nextIt, doneLabel, BRANCH_ANY_N_CLEAR);

    // the remaining instructions of the original block
    auto doneIt = method.emplaceLabel(nextIt, std::make_unique<BranchLabel>(*doneLabel));
    static_cast<void>(doneIt);
}

/*
 * Skips the block (to the given next block) if none of its lanes is active.
 */
static void insertBlockGuard(Method& method, BasicBlock& block, const Value& mask, const Local* nextLabel)
{
    auto it = block.walk().nextInBlock();
    auto activeLabel = createLabel(method, block.getLabel()->getLabel()->name.substr(1) + ".simt_active");
    assign(it, NOP_REGISTER) = (mask, SetFlag::SET_FLAGS);
    insertBranch(it, nextLabel, BRANCH_ALL_Z_SET);
    insertBranch(it, activeLabel, BRANCH_ANY_Z_CLEAR);
    auto activeIt = method.emplaceLabel(it, std::make_unique<BranchLabel>(*activeLabel));
    static_cast<void>(activeIt);
}

static void linearizeRegion(Method& method, const ControlFlow& cf, const Region& region, const Value& entryMask,
    FastMap<const IntermediateInstruction*, Value>& scatteredAccesses)
{
    FastMap<const BasicBlock*, Value> masks;
    for(auto b : region.blocks)
        masks.emplace(cf.blocks[b], method.addNewLocal(getMaskType(), "%simt_mask"));
    // loops: the lanes taking the back edge, and the header of every latch
    FastMap<const BasicBlock*, Value> continueMasks;
    FastMap<std::size_t, std::size_t> loopHeaders;
    for(const auto& loop : region.loops)
    {
        continueMasks.emplace(cf.blocks[loop.first], method.addNewLocal(getMaskType(), "%simt_continue_mask"));
        loopHeaders.emplace(loop.second, loop.first);
    }
    auto resolverFor = [&](std::size_t source) -> EdgeMaskResolver {
        return [&, source](const Edge& edge) -> const Value* {
            auto target = cf.indices.at(edge.target);
            auto& targetMasks = target <= source ? continueMasks : masks;
            auto it = targetMasks.find(edge.target);
            return it != targetMasks.end() ? &it->second : nullptr;
        };
    };
    auto localBlocks = determineLocalBlocks(method);

    // 1. the entry block (run by all lanes): initialize the masks and set the masks of its successors instead of
    // branching
    {
        const auto& exit = cf.exits[region.entryBlock];
        auto it = exit.instructions.empty() ? cf.blocks[region.entryBlock]->walkEnd() : exit.instructions.front();
        for(auto b : region.blocks)
            assign(it, masks.at(cf.blocks[b])) = BOOL_FALSE;
        for(const auto& entry : continueMasks)
            assign(it, entry.second) = BOOL_FALSE;
        insertEdgeMasks(method, it, exit, entryMask, resolverFor(region.entryBlock), {});
        for(auto instrIt : exit.instructions)
            instrIt.erase();
    }

    // 2. the blocks of the region, run one after the other
    for(std::size_t i = 0; i < region.blocks.size(); ++i)
    {
        auto& block = *cf.blocks[region.blocks[i]];
        const auto& exit = cf.exits[region.blocks[i]];
        const auto& mask = masks.at(&block);
        auto nextBlock = cf.blocks[i + 1 < region.blocks.size() ? region.blocks[i + 1] : region.mergeBlock];

        // a) values used outside of the block are written to temporaries
        FastSet<const IntermediateInstruction*> exitInstructions;
        for(const auto& instrIt : exit.instructions)
            exitInstructions.emplace(instrIt.get());
        FastMap<const Local*, const Local*> renamed;
        // the stores (false) and the accesses lane by lane (true)
        std::vector<std::pair<InstructionWalker, bool>> accesses;
        for(auto it = block.walk().nextInBlock(); !it.isEndOfBlock(); it.nextInBlock())
        {
            if(!it.has() || exitInstructions.find(it.get()) != exitInstructions.end())
                continue;
            for(const auto& entry : renamed)
            {
                if(it->readsLocal(entry.first))
                    it->replaceLocal(entry.first, entry.second, LocalUse::Type::READER);
            }
            bool isScattered = scatteredAccesses.find(it.get()) != scatteredAccesses.end();
            if(isMemoryWrite(*it.get()))
            {
                auto store = it.get<MemoryInstruction>();
                if(isScattered || store->getDestinationElementType().isVectorType())
                    accesses.emplace_back(it, isScattered);
                continue;
            }
            if(isScattered)
                // the loaded value is renamed below like any other value
                accesses.emplace_back(it, true);
            auto out = it->checkOutputLocal();
            if(!out || out->type.isLabelType() || !isUsedOutside(localBlocks, out, block))
                continue;
            auto renamedIt = renamed.find(out);
            if(renamedIt == renamed.end())
            {
                auto tmp = method.addNewLocal(out->type, out->name, "simt_masked").local();
                if(it->hasConditionalExecution())
                {
                    // the lanes not written here keep the previous value
                    it.emplace(std::make_unique<MoveOperation>(tmp->createReference(), out->createReference()));
                    it.nextInBlock();
                }
                renamedIt = renamed.emplace(out, tmp).first;
            }
            it->replaceLocal(out, renamedIt->second, LocalUse::Type::WRITER);
        }

        // b) at the end of the block: write the temporaries for the active lanes, set the masks of the successors and
        // remove the branches
        auto endIt = exit.instructions.empty() ? block.walkEnd() : exit.instructions.front();
        if(!renamed.empty())
        {
            assign(endIt, NOP_REGISTER) = (mask, SetFlag::SET_FLAGS);
            for(const auto& entry : renamed)
                assign(endIt, entry.first->createReference()) = (entry.second->createReference(), COND_ZERO_CLEAR);
        }
        insertEdgeMasks(method, endIt, exit, mask, resolverFor(region.blocks[i]), renamed);
        auto headerIt = loopHeaders.find(region.blocks[i]);
        if(headerIt != loopHeaders.end())
        {
            // the latch of a loop: run the loop again for the lanes taking the back edge, while there are any. The
            // masks of the loop's blocks start empty again.
            auto header = headerIt->second;
            const auto& headerMask = masks.at(cf.blocks[header]);
            const auto& continueMask = continueMasks.at(cf.blocks[header]);
            assign(endIt, headerMask) = continueMask;
            assign(endIt, continueMask) = BOOL_FALSE;
            for(auto b = header + 1; b <= region.blocks[i]; ++b)
                assign(endIt, masks.at(cf.blocks[b])) = BOOL_FALSE;
            assign(endIt, NOP_REGISTER) = (headerMask, SetFlag::SET_FLAGS);
            insertBranch(endIt, cf.blocks[header]->getLabel()->getLabel(), BRANCH_ANY_Z_CLEAR);
            insertBranch(endIt, nextBlock->getLabel()->getLabel(), BRANCH_ALL_Z_SET);
        }
        for(auto instrIt : exit.instructions)
            instrIt.erase();

        // c) stores only write the active lanes, accesses lane by lane only access them. This splits the block, so
        // start with the last access.
        for(auto accessIt = accesses.rbegin(); accessIt != accesses.rend(); ++accessIt)
        {
            if(accessIt->second)
            {
                auto addresses = scatteredAccesses.at(accessIt->first.get());
                scatteredAccesses.erase(accessIt->first.get());
                insertScatteredAccess(method, accessIt->first, mask, addresses);
            }
            else
                insertMaskedStore(method, accessIt->first, mask, entryMask);
        }

        // d) skip the block if no lane is active
        insertBlockGuard(method, block, mask, nextBlock->getLabel()->getLabel());
    }
}

/*
 * Runs the divergent regions of the converted kernel one block after the other for the lanes reaching each block.
 */
static void linearizeDivergentControlFlow(
    Method& method, FastMap<const IntermediateInstruction*, Value>& scatteredAccesses)
{
    auto cf = determineControlFlow(method);
    std::vector<Region> regions;
    // after the conversion, all conditions differing between lanes are vectors
    auto error = findDivergentRegions(
        cf, [](const Value& cond) -> bool { return cond.type.isVectorType(); }, regions);
    if(!error.empty())
        throw CompilationError(CompilationStep::NORMALIZER, "SIMT: unexpected divergent control flow", error);
    if(regions.empty() && scatteredAccesses.empty())
        return;

    auto entryMask = insertEntryMask(method);
    for(const auto& region : regions)
    {
        CPPLOG_LAZY(logging::Level::DEBUG,
            log << "SIMT: linearizing divergent region from " << cf.blocks[region.entryBlock]->to_string() << " to "
                << cf.blocks[region.mergeBlock]->to_string() << " with " << region.blocks.size() << " blocks"
                << logging::endl);
        linearizeRegion(method, cf, region, entryMask, scatteredAccesses);
    }

    // the accesses lane by lane outside of divergent regions access all lanes of the work-group
    std::vector<InstructionWalker> remainingAccesses;
    for(auto& block : method)
    {
        for(auto it = block.walk(); !it.isEndOfBlock(); it.nextInBlock())
        {
            if(it.has() && scatteredAccesses.find(it.get()) != scatteredAccesses.end())
                remainingAccesses.push_back(it);
        }
    }
    // splitting the blocks keeps the walkers of the instructions before the split position valid
    for(auto it = remainingAccesses.rbegin(); it != remainingAccesses.rend(); ++it)
    {
        auto addresses = scatteredAccesses.at(it->get());
        scatteredAccesses.erase(it->get());
        insertScatteredAccess(method, *it, entryMask, addresses);
    }
    CPPLOG_LAZY(logging::Level::DEBUG, log << "SIMT: kernel after linearization:" << logging::endl);
    method.dumpInstructions();
}

/*
 * Lays out the blocks in a topological order of the forward edges which keeps the blocks of every loop together (as
 * e.g. in WebAssembly's CFG sort): every block follows all its predecessors except for loop back edges, and a loop's
 * blocks follow its header without other blocks in between. Then the blocks of a divergent region lie between its
 * entry and its merge point. Clang sometimes places e.g. the body of an if after the following block, which looks
 * like a loop entered in the middle.
 *
 * Returns whether the order changed. Does nothing for irreducible control flow (loops with several entries).
 */
static bool reorderBlocks(Method& method)
{
    std::vector<BasicBlock*> blocks;
    FastMap<const BasicBlock*, std::size_t> indices;
    for(auto& block : method)
    {
        indices.emplace(&block, blocks.size());
        blocks.push_back(&block);
    }
    const auto numBlocks = blocks.size();
    // the successors (also by falling through to the next block) and the predecessors
    std::vector<std::vector<std::size_t>> successors(numBlocks);
    std::vector<std::vector<std::size_t>> predecessors(numBlocks);
    std::vector<bool> fallsThrough(numBlocks, false);
    for(std::size_t b = 0; b < numBlocks; ++b)
    {
        auto addEdge = [&](std::size_t target) {
            if(std::find(successors[b].begin(), successors[b].end(), target) != successors[b].end())
                return;
            successors[b].push_back(target);
            predecessors[target].push_back(b);
        };
        bool returns = false;
        for(const auto& instr : *blocks[b])
        {
            if(auto branch = dynamic_cast<const Branch*>(instr.get()))
            {
                for(auto target : branch->getTargetLabels())
                {
                    auto targetBlock = method.findBasicBlock(target);
                    if(!targetBlock)
                        // e.g. a computed branch
                        return false;
                    addEdge(indices.at(targetBlock));
                }
            }
            else if(dynamic_cast<const Return*>(instr.get()))
                returns = true;
        }
        if(!returns && b + 1 < numBlocks && blocks[b]->fallsThroughToNextBlock(false))
        {
            fallsThrough[b] = true;
            addEdge(b + 1);
        }
    }

    // back edges (to a block on the DFS stack) and the loops they form
    std::vector<int> state(numBlocks, 0); // 0 = not visited, 1 = on stack, 2 = done
    std::vector<std::pair<std::size_t, std::size_t>> backEdges;
    {
        std::vector<std::pair<std::size_t, std::size_t>> stack{{0, 0}};
        state[0] = 1;
        while(!stack.empty())
        {
            auto& top = stack.back();
            if(top.second < successors[top.first].size())
            {
                auto next = successors[top.first][top.second++];
                if(state[next] == 1)
                    backEdges.emplace_back(top.first, next);
                else if(state[next] == 0)
                {
                    state[next] = 1;
                    stack.emplace_back(next, 0);
                }
            }
            else
            {
                state[top.first] = 2;
                stack.pop_back();
            }
        }
    }
    // the blocks of the loop of every header: the header and all blocks reaching a latch without passing the header
    FastMap<std::size_t, FastSet<std::size_t>> loops;
    for(const auto& edge : backEdges)
    {
        auto& body = loops[edge.second];
        body.emplace(edge.second);
        std::vector<std::size_t> pending{edge.first};
        while(!pending.empty())
        {
            auto current = pending.back();
            pending.pop_back();
            if(!body.emplace(current).second)
                continue;
            for(auto pred : predecessors[current])
                pending.push_back(pred);
        }
    }
    std::set<std::pair<std::size_t, std::size_t>> backEdgeSet(backEdges.begin(), backEdges.end());
    for(const auto& loop : loops)
    {
        // a single entry: only the header is entered from outside the loop
        for(auto b : loop.second)
        {
            if(b == loop.first)
                continue;
            for(auto pred : predecessors[b])
            {
                if(loop.second.find(pred) == loop.second.end())
                    return false;
            }
        }
    }

    // topological order of the forward edges, with the blocks of the innermost loop of the last placed header first
    std::vector<std::size_t> remainingPredecessors(numBlocks, 0);
    for(std::size_t b = 0; b < numBlocks; ++b)
    {
        for(auto pred : predecessors[b])
        {
            if(state[pred] != 0 && backEdgeSet.find(std::make_pair(pred, b)) == backEdgeSet.end())
                ++remainingPredecessors[b];
        }
    }
    std::vector<std::size_t> order;
    std::vector<bool> placed(numBlocks, false);
    std::vector<std::size_t> openLoops;
    std::set<std::size_t> ready{0};
    while(!ready.empty())
    {
        // close the loops whose blocks are all placed
        while(!openLoops.empty())
        {
            const auto& body = loops.at(openLoops.back());
            if(std::any_of(body.begin(), body.end(), [&](std::size_t b) { return !placed[b]; }))
                break;
            openLoops.pop_back();
        }
        // the first ready block (in the original order) in the innermost open loop
        auto next = ready.end();
        for(auto it = ready.begin(); it != ready.end(); ++it)
        {
            if(openLoops.empty() || loops.at(openLoops.back()).count(*it))
            {
                next = it;
                break;
            }
        }
        if(next == ready.end())
            // should not happen for single-entry loops
            return false;
        auto block = *next;
        ready.erase(next);
        placed[block] = true;
        order.push_back(block);
        if(loops.find(block) != loops.end())
            openLoops.push_back(block);
        for(auto succ : successors[block])
        {
            if(backEdgeSet.find(std::make_pair(block, succ)) != backEdgeSet.end())
                continue;
            if(--remainingPredecessors[succ] == 0)
                ready.emplace(succ);
        }
    }
    // unreachable blocks stay at the end
    for(std::size_t b = 0; b < numBlocks; ++b)
    {
        if(!placed[b])
            order.push_back(b);
    }
    // the default last block stays the last one (e.g. the work-group loop continues after it)
    auto lastIt = std::find_if(order.begin(), order.end(), [&](std::size_t b) -> bool {
        auto label = blocks[b]->getLabel();
        return label && label->getLabel()->name == BasicBlock::LAST_BLOCK;
    });
    if(lastIt != order.end())
    {
        auto last = *lastIt;
        order.erase(lastIt);
        order.push_back(last);
    }
    bool changed = false;
    for(std::size_t i = 0; i < numBlocks; ++i)
        changed = changed || order[i] != i;
    if(!changed)
        return false;

    // make falling through explicit, since the next block changes
    for(std::size_t b = 0; b < numBlocks; ++b)
    {
        if(fallsThrough[b])
            blocks[b]->walkEnd().emplace(std::make_unique<Branch>(blocks[b + 1]->getLabel()->getLabel()));
    }
    for(auto b : order)
    {
        auto it = method.begin();
        while(&*it != blocks[b])
            ++it;
        method.moveBlock(it, method.end());
    }
    CPPLOG_LAZY(logging::Level::DEBUG, log << "SIMT: reordered the blocks of the kernel:" << logging::endl);
    method.dumpInstructions();
    return true;
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

    CPPLOG_LAZY(logging::Level::DEBUG, log << "SIMT: checking kernel '" << method.name << "':" << logging::endl);
    method.dumpInstructions();

    std::string vectorTypes;
    auto vectorWidth = determineVectorWidth(method, vectorTypes);
    auto reason = vectorWidth ? checkUnsupported(method, vectorWidth, normalization::isWorkItemLoopsEnabled(config)) :
                                "unsupported vector types:" + vectorTypes;
    if(reason.empty())
    {
        auto cf = determineControlFlow(method);
        auto analysis = analyzeVaryingValues(method, cf, vectorWidth);
        CPPLOG_LAZY_BLOCK(logging::Level::DEBUG, {
            for(const auto& entry : analysis.varying)
            {
                auto writer = entry.first->getSingleWriter();
                logging::debug() << "SIMT: work-item dependent " << entry.first->to_string() << ", lane stride "
                                 << (entry.second.stride ? std::to_string(*entry.second.stride) : "unknown")
                                 << ", written by: " << (writer ? writer->to_string() : "(several)") << logging::endl;
            }
        });
        std::vector<Region> regions;
        if(analysis.varying.empty())
            reason = "does not use work-item IDs";
        else
            reason = findDivergentRegions(cf, [&](const Value& cond) { return analysis.isVarying(cond); }, regions);
        // the original order of the blocks, restored if the kernel isn't converted after all
        std::vector<BasicBlock*> originalOrder;
        for(auto& block : method)
            originalOrder.push_back(&block);
        bool reordered = false;
        if(!reason.empty() && !analysis.varying.empty() && std::getenv("VC4C_NO_BLOCK_REORDER") == nullptr &&
            reorderBlocks(method))
        {
            reordered = true;
            // the blocks of the divergent regions might not have been laid out between their entry and merge point
            CPPLOG_LAZY(logging::Level::DEBUG,
                log << "SIMT: retrying with the blocks reordered after: " << reason << logging::endl);
            cf = determineControlFlow(method);
            analysis = analyzeVaryingValues(method, cf, vectorWidth);
            regions.clear();
            reason = findDivergentRegions(cf, [&](const Value& cond) { return analysis.isVarying(cond); }, regions);
        }
        if(reason.empty())
            reason = checkVaryingValues(method, analysis, cf, regions, vectorWidth);
        if(reason.empty())
        {
            FastMap<const IntermediateInstruction*, Value> scatteredAccesses;
            convertToSIMT(method, analysis, vectorWidth, scatteredAccesses);
            linearizeDivergentControlFlow(method, scatteredAccesses);
            CPPLOG_LAZY(logging::Level::INFO,
                log << "SIMT: running kernel '" << method.name << "' with "
                    << (vectorWidth > 1 ? std::to_string(vectorWidth) + " SIMD lanes per work-item" :
                                          std::string("one work-item per SIMD lane"))
                    << " (" << analysis.varying.size() << " work-item dependent values)" << logging::endl);
            return;
        }
        if(reordered)
        {
            // the kernel is not converted, keep its blocks as they were (the falls-through are explicit branches now)
            for(auto block : originalOrder)
            {
                auto it = method.begin();
                while(&*it != block)
                    ++it;
                method.moveBlock(it, method.end());
            }
        }
    }
    CPPLOG_LAZY(logging::Level::INFO,
        log << "SIMT: not using SIMT mode for kernel '" << method.name << "': " << reason << logging::endl);
}
