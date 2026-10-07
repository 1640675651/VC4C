/*
 * Author: doe300
 *
 * See the file "LICENSE" for the full license governing this code.
 */
#ifndef VC4C_NORMALIZATION_WORK_ITEM_LOOPS_H
#define VC4C_NORMALIZATION_WORK_ITEM_LOOPS_H

namespace vc4c
{
    class Method;
    class Module;
    struct Configuration;

    namespace normalization
    {
        // The name of the optimization pass, which can be disabled via --fno-work-item-loops
        extern const char* const WORK_ITEM_LOOPS_PASS_NAME;

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
         * Supported are kernels without private arrays (stack allocations), with at most 16 such values per barrier,
         * all of them scalars of at most 32 bits or pointers, not calling work-group functions with their own
         * barriers (e.g. async_work_group_copy). Other kernels are not changed and keep the limit of one work-item per
         * QPU.
         *
         * NOTE: Needs to run before the work-item functions and barriers are intrinsified.
         */
        void loopWorkItems(Module& module, Method& method, const Configuration& config);
    } // namespace normalization
} // namespace vc4c

#endif /* VC4C_NORMALIZATION_WORK_ITEM_LOOPS_H */
