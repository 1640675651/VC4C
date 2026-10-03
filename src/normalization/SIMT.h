/*
 * See the file "LICENSE" for the full license governing this code.
 */
#ifndef VC4C_NORMALIZATION_SIMT_H
#define VC4C_NORMALIZATION_SIMT_H

namespace vc4c
{
    class Method;
    class Module;
    struct Configuration;

    namespace normalization
    {
        /*
         * The name to switch SIMT mode on or off like an optimization pass: --fsimt, --fno-simt (also as OpenCL build
         * options in VC4CL). These take precedence over the environment variable VC4C_NO_SIMT.
         */
        constexpr const char* SIMT_PASS_NAME = "simt";

        /*
         * Runs the kernel in "SIMT mode" if possible: one work-item per SIMD lane, 16 work-items per QPU (see
         * doc/SIMT.md).
         *
         * Every value which differs between work-items is converted into a 16-element vector (one element per
         * work-item), all other values stay as they are. Divergent branches and loops are linearized and run
         * with masks of active lanes. Only kernels whose memory accesses with work-item dependent addresses are
         * contiguous across the work-items are converted. Other kernels are left unchanged.
         *
         * Needs to run before the work-item functions are intrinsified. Can be disabled with --fno-simt or by setting
         * the environment variable VC4C_NO_SIMT.
         */
        void vectorizeWorkItems(Module& module, Method& method, const Configuration& config);
    } // namespace normalization
} // namespace vc4c

#endif /* VC4C_NORMALIZATION_SIMT_H */
