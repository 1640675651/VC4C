/*
 * Author: doe300
 *
 * See the file "LICENSE" for the full license governing this code.
 */
#ifndef VC4C_NORMALIZATION_PRINTF_H
#define VC4C_NORMALIZATION_PRINTF_H

namespace vc4c
{
    class Method;
    class Module;
    class InstructionWalker;
    struct Configuration;

    namespace normalization
    {
        /*
         * Lowers calls to printf() into writing a record into the printf buffer, a hidden last kernel parameter (see
         * PRINTF_BUFFER_PARAMETER_NAME), which the run-time formats and prints after the kernel finished.
         *
         * Buffer layout (32-bit words):
         * | bytes of records written | records ... (PRINTF_BUFFER_SIZE bytes) | spill area (PRINTF_MAX_RECORD_SIZE) |
         *
         * Record layout (32-bit words):
         * | size of the record in bytes | address of the global data | address of the format string | arguments ... |
         *
         * The format string (and the strings printed with %s) are in the global data, the run-time maps them back to
         * the program's global data via the address of the global data.
         *
         * Every scalar argument takes one word (integers zero-extended, floats as float, pointers as address), except
         * 64-bit integers which take two words (lower and upper word), every vector argument the words of its elements.
         * Float arguments promoted to double are written as float. A record not fitting into the buffer anymore is written into the spill
         * area (and not printed), and the call returns -1.
         *
         * NOTE: Needs to run before the intrinsics, which cannot handle the float arguments promoted to double.
         */
        void lowerPrintf(Module& module, Method& method, InstructionWalker it, const Configuration& config);
    } // namespace normalization
} // namespace vc4c

#endif /* VC4C_NORMALIZATION_PRINTF_H */
