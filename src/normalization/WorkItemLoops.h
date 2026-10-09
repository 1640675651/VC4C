/*
 * Author: doe300
 *
 * See the file "LICENSE" for the full license governing this code.
 */
#ifndef VC4C_NORMALIZATION_WORK_ITEM_LOOPS_H
#define VC4C_NORMALIZATION_WORK_ITEM_LOOPS_H

#include <cstdint>
#include <string>

namespace vc4c
{
    class Method;
    class Module;
    struct Configuration;
    struct Global;
    struct Parameter;

    namespace normalization
    {
        // The name of the optimization pass, which can be disabled via --fno-work-item-loops
        extern const char* const WORK_ITEM_LOOPS_PASS_NAME;

        // Whether the pass is enabled (also disabled by the environment variable VC4C_NO_WORK_ITEM_LOOPS)
        bool isWorkItemLoopsEnabled(const Configuration& config);

        // The names (without the "%") of the hidden parameters of kernels whose work-groups run on teams of QPUs at the
        // same time (see loopWorkItems), set by the run-time: the QPU's team, and the team's copy of the kernel's
        // __local variables (each starting at a multiple of LOCAL_VARIABLE_ALIGNMENT bytes)
        constexpr const char* WORK_GROUP_TEAM_PARAMETER_NAME = "__vc4cl_work_group_team";
        constexpr const char* LOCAL_VARIABLES_PARAMETER_NAME = "__vc4cl_local_variables";
        constexpr uint32_t LOCAL_VARIABLE_ALIGNMENT = 16;

        // The hidden parameter with the given name (without the "%"), if any
        const Parameter* findHiddenParameter(const Method& method, const std::string& name);

        // The bytes the __local variable takes in the __local variables parameter
        uint32_t getLocalVariableSize(const Global& global);

        /*
         * Lets kernels with barriers run work-groups of more work-items than QPUs.
         *
         * Without this, every work-item of a kernel with barriers runs on its own QPU, all at the same time, so a
         * work-group has at most 12 work-items. Instead, the QPUs of a work-group (at most 12) loop over their
         * work-items: work-item i runs on the QPU i modulo the number of QPUs. The kernel is split into regions at the
         * barriers, and every region is a loop over the QPU's work-items. Between the regions, the QPUs of the
         * work-group synchronize with the semaphore barrier. Values computed before and used after a barrier, which
         * differ between the work-items, are kept per work-item in the lanes of a vector register (lane j for the j-th
         * work-item of the QPU), so a QPU runs up to 16 work-items, a work-group up to 192.
         *
         * Kernels with __local memory, but without barriers, loop over their work-items in a single region (kernels
         * with neither run their work-items independently anyway).
         *
         * Barriers may be in loops and branches, as long as these are work-group uniform (as required by OpenCL): a
         * region then ends at the next barrier reached, and the loop over the work-items continues with the region
         * the QPU is in.
         *
         * Up to 8 such values (scalars of at most 32 bits or pointers) are kept in registers, all others (also
         * vectors) in a private variable in the work-item's stack frame in RAM. Private memory (stack allocations) is
         * per work-item: every work-item has its own stack frame. 64-bit values are kept as their two 32-bit words.
         *
         * In SIMT mode (see normalization/SIMT.cpp), the QPUs loop over the chunks of up to 16 work-items of their
         * work-group instead (the lanes of the QPU's registers). All values kept per chunk across barriers are vectors,
         * kept in RAM.
         *
         * NOTE: Needs to run after the SIMT conversion and before the work-item functions and barriers are
         * intrinsified.
         */
        void loopWorkItems(Module& module, Method& method, const Configuration& config);

        /*
         * Replaces private variables (stack allocations) of a scalar or vector type, which are only loaded and stored
         * as a whole, by locals. Clang keeps some of them (e.g. with life-time markers), which VC4C would lower into
         * registers later, but which the SIMT conversion and the work-item loops would otherwise need to keep in
         * memory (per work-item).
         */
        void promoteSimpleStackAllocations(Method& method);
    } // namespace normalization
} // namespace vc4c

#endif /* VC4C_NORMALIZATION_WORK_ITEM_LOOPS_H */
