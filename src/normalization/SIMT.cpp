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
    if(sizes[0] != 0 && (sizes[0] > NUM_QPUS * SIMT_WIDTH || sizes[1] > 1 || sizes[2] > 1))
        return "required work-group size is larger than 192 or not 1-dimensional";
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
 * from the entry before reaching the merge block. If the divergent branch is inside a loop, the lanes may leave the loop
 * in different iterations, so the region grows to contain the whole loop: its entry is then the block before the loop
 * header. Loops in the region need to consist of consecutive blocks with the header first and the only back edge from
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
static Analysis analyzeVaryingValues(Method& method, const ControlFlow& cf)
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
static std::string checkVaryingValues(
    const Method& method, const Analysis& analysis, const ControlFlow& cf, const std::vector<Region>& regions)
{
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
                    if(mem->op == MemoryOperation::WRITE && maskedBlocks.find(&block) != maskedBlocks.end() &&
                        elementType.getScalarBitCount() != 32)
                        // see insertMaskedStore
                        return "store of 8- or 16-bit values in divergent code: " + mem->to_string();
                }
                else if(mem->op == MemoryOperation::WRITE && analysis.isVarying(mem->getSource()))
                    return "work-items writing different values to the same address: " + mem->to_string();
            }
            if(readsVarying && !dynamic_cast<const Operation*>(instr.get()) &&
                !dynamic_cast<const MoveOperation*>(instr.get()) &&
                !dynamic_cast<const IntrinsicOperation*>(instr.get()) &&
                !dynamic_cast<const MemoryInstruction*>(instr.get()) && !dynamic_cast<const MethodCall*>(instr.get()) &&
                !dynamic_cast<const Branch*>(instr.get()))
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

    // 5. memory stores only write back the elements of work-items in the work-group, which needs the local size and
    // the QPU's first local ID (see lowerWriteRAM in periphery/VPM.cpp). Read them here, so the UNIFORMs are loaded at
    // the start of the kernel.
    auto startIt = method.walkAllInstructions();
    if(!startIt.isEndOfMethod() && startIt.get<BranchLabel>())
        startIt.nextInBlock();
    for(auto type : {BuiltinLocal::Type::LOCAL_SIZES, BuiltinLocal::Type::LOCAL_IDS})
    {
        auto builtin = method.findOrCreateBuiltin(type);
        startIt.emplace(std::make_unique<MoveOperation>(
            method.addNewLocal(TYPE_INT32, "%simt" + builtin->name.substr(1)), builtin->createReference()));
        startIt.nextInBlock();
    }

    method.metaData.mergedWorkItemsFactor = SIMT_WIDTH;
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

static void linearizeRegion(Method& method, const ControlFlow& cf, const Region& region, const Value& entryMask)
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
        std::vector<InstructionWalker> stores;
        for(auto it = block.walk().nextInBlock(); !it.isEndOfBlock(); it.nextInBlock())
        {
            if(!it.has() || exitInstructions.find(it.get()) != exitInstructions.end())
                continue;
            for(const auto& entry : renamed)
            {
                if(it->readsLocal(entry.first))
                    it->replaceLocal(entry.first, entry.second, LocalUse::Type::READER);
            }
            if(isMemoryWrite(*it.get()))
            {
                auto store = it.get<MemoryInstruction>();
                if(store->getDestinationElementType().isVectorType())
                    stores.push_back(it);
                continue;
            }
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

        // c) stores only write the active lanes. This splits the block, so start with the last store.
        for(auto storeIt = stores.rbegin(); storeIt != stores.rend(); ++storeIt)
            insertMaskedStore(method, *storeIt, mask, entryMask);

        // d) skip the block if no lane is active
        insertBlockGuard(method, block, mask, nextBlock->getLabel()->getLabel());
    }
}

/*
 * Runs the divergent regions of the converted kernel one block after the other for the lanes reaching each block.
 */
static void linearizeDivergentControlFlow(Method& method)
{
    auto cf = determineControlFlow(method);
    std::vector<Region> regions;
    // after the conversion, all conditions differing between lanes are vectors
    auto error = findDivergentRegions(
        cf, [](const Value& cond) -> bool { return cond.type.isVectorType(); }, regions);
    if(!error.empty())
        throw CompilationError(CompilationStep::NORMALIZER, "SIMT: unexpected divergent control flow", error);
    if(regions.empty())
        return;

    auto entryMask = insertEntryMask(method);
    for(const auto& region : regions)
    {
        CPPLOG_LAZY(logging::Level::DEBUG,
            log << "SIMT: linearizing divergent region from " << cf.blocks[region.entryBlock]->to_string() << " to "
                << cf.blocks[region.mergeBlock]->to_string() << " with " << region.blocks.size() << " blocks"
                << logging::endl);
        linearizeRegion(method, cf, region, entryMask);
    }
    CPPLOG_LAZY(logging::Level::DEBUG, log << "SIMT: kernel after linearization:" << logging::endl);
    method.dumpInstructions();
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

    auto reason = checkUnsupported(method);
    if(reason.empty())
    {
        auto cf = determineControlFlow(method);
        auto analysis = analyzeVaryingValues(method, cf);
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
        if(reason.empty())
            reason = checkVaryingValues(method, analysis, cf, regions);
        if(reason.empty())
        {
            convertToSIMT(method, analysis);
            linearizeDivergentControlFlow(method);
            CPPLOG_LAZY(logging::Level::INFO,
                log << "SIMT: running kernel '" << method.name << "' with one work-item per SIMD lane ("
                    << analysis.varying.size() << " work-item dependent values)" << logging::endl);
            return;
        }
    }
    CPPLOG_LAZY(logging::Level::INFO,
        log << "SIMT: not using SIMT mode for kernel '" << method.name << "': " << reason << logging::endl);
}
