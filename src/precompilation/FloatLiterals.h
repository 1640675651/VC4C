/*
 * Author: doe300
 *
 * See the file "LICENSE" for the full license governing this code.
 */
#ifndef VC4C_PRECOMPILATION_FLOAT_LITERALS_H
#define VC4C_PRECOMPILATION_FLOAT_LITERALS_H

#include <string>

namespace vc4c
{
    namespace precompilation
    {
        /*
         * Rewrites the floating-point literals of the given OpenCL C source to be converted rounding toward zero, the
         * rounding mode of the VideoCore IV GPU, instead of to nearest as by clang.
         *
         * C99 (6.4.4.2), which OpenCL C is based on, allows the conversion to either neighbor of the nearest value.
         * Rounding toward zero makes the constants consistent with the calculations on the device, and OpenCL-CTS
         * (e.g. test_printf) expects this for devices only supporting this rounding mode. Since clang ignores
         * "#pragma STDC FENV_ROUND" for SPIR, the literals are replaced with the hexadecimal literals of the values
         * rounded toward zero (the suffix is kept).
         *
         * Only literals whose value differs when rounding toward zero are replaced. Literals overflowing or
         * underflowing are kept, as are literals in comments, string and character literals and in #include, #pragma,
         * #line, #error and #warning directives.
         *
         * Returns whether any literal was replaced, the rewritten source is written to result.
         */
        bool roundFloatLiteralsTowardZero(const std::string& source, std::string& result);
    } // namespace precompilation
} // namespace vc4c

#endif /* VC4C_PRECOMPILATION_FLOAT_LITERALS_H */
