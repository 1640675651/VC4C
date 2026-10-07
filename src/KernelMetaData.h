/*
 * Author: doe300
 *
 * See the file "LICENSE" for the full license governing this code.
 */

#ifndef VC4C_KERNEL_METADATA_H
#define VC4C_KERNEL_METADATA_H

#include "config.h"
#include "shared/BinaryHeader.h"

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>

namespace vc4c
{
    class Local;

    /**
     * Container for additional meta-data of kernel-functions
     */
    struct KernelMetaData
    {
        /**
         * The implicit UNIFORMs actually used
         */
        KernelUniforms uniformsUsed;
        /**
         * The compilation-time work-group size, specified by the reqd_work_group_size attribute
         */
        std::array<uint32_t, 3> workGroupSizes;
        /**
         * The compilation-time preferred work-group size, specified by the work_group_size_hint attribute
         */
        std::array<uint32_t, 3> workGroupSizeHints;
        /**
         * The factor with which the work-items are merged, e.g. 16 if 16 work-items are merged into one QPU execution.
         */
        uint8_t mergedWorkItemsFactor;
        /**
         * For kernels looping over the work-items of their QPU (see normalization/WorkItemLoops.cpp): the local
         * holding the (packed) local IDs of the current work-item, which the work-item functions read instead of the
         * UNIFORM. NULL for other kernels.
         */
        const Local* workItemLoopLocalIds = nullptr;
        /**
         * The meta data strings to be stored in the module and used by the run-time for CL_KERNEL_ATTRIBUTES queries.
         */
        std::vector<MetaData> entries;

        KernelMetaData() : uniformsUsed(), workGroupSizes(), workGroupSizeHints(), mergedWorkItemsFactor(0)
        {
            workGroupSizes.fill(0);
            workGroupSizeHints.fill(0);
        }

        /**
         * Returns the explicit work-group size, if it is set
         */
        inline Optional<uint32_t> getFixedWorkGroupSize() const
        {
            if(std::any_of(workGroupSizes.begin(), workGroupSizes.end(), [](uint32_t u) -> bool { return u > 0; }))
                return std::accumulate(workGroupSizes.begin(), workGroupSizes.end(), 1u, std::multiplies<uint32_t>{});
            return {};
        }

        /**
         * Returns the maximum number of work-items in a work-group for this kernel
         */
        inline uint32_t getMaximumWorkGroupSize() const
        {
            if(auto fixedSize = getFixedWorkGroupSize())
                return *fixedSize;
            // Work-groups of kernels with independent work-items (always in SIMT mode) are split into chunks of
            // work-items run by any QPU, so they may have up to 16 work-items per QPU. Only kernels with barriers or
            // __local memory are limited to one work-item per QPU, but this is only known after normalization.
            return NUM_QPUS * NATIVE_VECTOR_SIZE;
        }

        /**
         * Returns the maximum number of kernel instances to be executed (the maximum number of QPUs required) for a
         * single work-group
         */
        inline uint32_t getMaximumInstancesCount() const
        {
            // With merged work-items (SIMT mode), up to NUM_QPUS work-groups run in parallel, one per QPU
            if(mergedWorkItemsFactor > 1)
                return NUM_QPUS;
            auto factor = std::max(mergedWorkItemsFactor, uint8_t{1});
            if(auto fixedSize = getFixedWorkGroupSize())
                // round up if the fixed number of work-items do not match exactly. At most all QPUs run at the same
                // time: larger work-groups run in chunks (independent work-items) or are rejected by VC4CL when
                // launched (barriers, __local memory). OpenCL requires such kernels to compile anyway.
                return std::min(NUM_QPUS, (*fixedSize / factor) + (*fixedSize % factor != 0));
            return NUM_QPUS;
        }
    };
} // namespace vc4c

#endif /* VC4C_KERNEL_METADATA_H */
