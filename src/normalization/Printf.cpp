/*
 * Author: doe300
 *
 * See the file "LICENSE" for the full license governing this code.
 */

#include "Printf.h"

#include "LongOperations.h"
#include "../InstructionWalker.h"
#include "../Method.h"
#include "../SIMDVector.h"
#include "../intermediate/Helper.h"
#include "../intermediate/VectorHelper.h"
#include "../intermediate/operators.h"
#include "config.h"
#include "log.h"

#include <string>
#include <vector>

using namespace vc4c;
using namespace vc4c::intermediate;
using namespace vc4c::normalization;
using namespace vc4c::operators;

static const std::string PRINTF_FUNCTION = "printf";

static const Parameter& getOrAddPrintfBuffer(Method& method)
{
    const std::string name = std::string("%") + PRINTF_BUFFER_PARAMETER_NAME;
    if(!method.parameters.empty() && method.parameters.back().name == name)
        return method.parameters.back();
    // the parameters are referred to by pointer, so they must not move, see BitcodeReader::parseFunction
    if(method.parameters.size() >= method.parameters.capacity())
        throw CompilationError(
            CompilationStep::NORMALIZER, "Cannot add the printf buffer parameter to kernel", method.name);
    return method.addParameter(Parameter(name, method.createPointerType(TYPE_INT32, AddressSpace::GLOBAL),
        add_flag(ParameterDecorations::INPUT, ParameterDecorations::OUTPUT)));
}

/*
 * Converts the value into the 32-bit words to be written into the record: integers of less than 32 bits are
 * zero-extended, 64-bit integers split into the lower and upper word, pointers converted to their address.
 */
static void appendWords(InstructionWalker& it, const Value& val, std::vector<Value>& words)
{
    if(val.type.getPointerType())
    {
        words.push_back(assign(it, TYPE_INT32, "%printf_word") = val);
        return;
    }
    if(val.type.isFloatingType())
    {
        if(val.type.getScalarBitCount() != 32)
            throw CompilationError(CompilationStep::NORMALIZER,
                "Only 32-bit floating-point values can be printed on this device", val.to_string());
        words.push_back(val);
        return;
    }
    auto bits = val.type.getScalarBitCount();
    if(bits > 32)
    {
        // the parts of the 64-bit local are written when lowering the 64-bit operations
        Value lower = UNDEFINED_VALUE;
        Value upper = UNDEFINED_VALUE;
        std::tie(lower, upper) = getLowerAndUpperWords(val);
        words.push_back(lower);
        words.push_back(upper);
    }
    else if(bits == 32)
        words.push_back(val);
    else
        words.push_back(assign(it, TYPE_INT32, "%printf_word") = (val & Value(Literal((1u << bits) - 1u), TYPE_INT32)));
}

/*
 * The float argument of a variadic call is promoted to double by the front-end, which this device doesn't support:
 * Returns the original float value.
 */
static Value getUnpromotedArgument(Method& method, const Value& arg, std::vector<const Local*>& removedPromotions)
{
    if(!arg.type.isFloatingType() || arg.type.getScalarBitCount() != 64)
        return arg;
    // constant (e.g. literal or folded) float arguments are promoted to double constants, which are stored as float
    if(auto lit = arg.getLiteralValue())
        return Value(*lit, TYPE_FLOAT);
    auto promotion = dynamic_cast<const IntrinsicOperation*>(arg.getSingleWriter());
    if(!promotion || promotion->opCode != "fpext" || promotion->getFirstArg().type.getScalarBitCount() != 32)
        throw CompilationError(
            CompilationStep::NORMALIZER, "Double values cannot be printed on this device", arg.to_string());
    removedPromotions.push_back(arg.checkLocal());
    return promotion->getFirstArg();
}

static void removeUnusedPromotions(Method& method, const std::vector<const Local*>& promotedValues)
{
    for(auto loc : promotedValues)
    {
        if(!loc || loc->hasUsers(LocalUse::Type::READER))
            continue;
        auto writer = loc->getSingleWriter();
        // replace the conversion to double, which the intrinsics would fail to lower. The instruction is not erased,
        // since the normalization might hold a walker to it (e.g. if it directly precedes the printf() call)
        auto it = method.walkAllInstructions();
        while(!it.isEndOfMethod())
        {
            if(it.get() == writer)
            {
                it.reset(std::make_unique<Nop>(DelayType::WAIT_REGISTER));
                break;
            }
            it.nextInMethod();
        }
    }
}

void normalization::lowerPrintf(Module& module, Method& method, InstructionWalker it, const Configuration& config)
{
    auto call = it.get<MethodCall>();
    if(!call || call->methodName != PRINTF_FUNCTION || call->getArguments().empty())
        return;

    CPPLOG_LAZY(logging::Level::DEBUG, log << "Lowering printf call: " << call->to_string() << logging::endl);
    const Value buffer = getOrAddPrintfBuffer(method).createReference();

    // 1. the words to write: the size of the record, the format string and the arguments
    std::vector<const Local*> promotedValues;
    std::vector<Value> argumentWords;
    const auto args = call->getArguments();
    for(std::size_t i = 1; i < args.size(); ++i)
    {
        auto arg = getUnpromotedArgument(method, args[i], promotedValues);
        if(arg.type.isVectorType())
        {
            // one word per element
            for(uint8_t e = 0; e < arg.type.getVectorWidth(); ++e)
            {
                if(auto constantVector = arg.checkVector())
                {
                    appendWords(it, Value((*constantVector)[e], arg.type.getElementType()), argumentWords);
                    continue;
                }
                auto element = method.addNewLocal(arg.type.getElementType(), "%printf_element");
                it = insertVectorExtraction(it, method, arg, Value(Literal(static_cast<uint32_t>(e)), TYPE_INT8), element);
                appendWords(it, element, argumentWords);
            }
        }
        else
            appendWords(it, arg, argumentWords);
    }
    const uint32_t recordSize = static_cast<uint32_t>((3 + argumentWords.size()) * sizeof(uint32_t));
    if(recordSize > PRINTF_MAX_RECORD_SIZE)
        throw CompilationError(CompilationStep::NORMALIZER, "Too many arguments for printf", call->to_string());
    const Value formatWord = assign(it, TYPE_INT32, "%printf_word") = args[0];

    // 2. reserve the space for the record (with the same mutex as the atomic functions)
    it.emplace(std::make_unique<MethodCall>("vc4cl_mutex_lock"));
    it.nextInBlock();
    auto used = method.addNewLocal(TYPE_INT32, "%printf_used");
    // the counter accesses must not lock the mutex themselves (as for the atomic functions)
    it.emplace(std::make_unique<MemoryInstruction>(
        MemoryOperation::READ, Value(used), Value(buffer), Value(INT_ONE), false /* already locked */));
    it.nextInBlock();
    auto newUsed = assign(it, TYPE_INT32, "%printf_used") = used + Value(Literal(recordSize), TYPE_INT32);
    // all conditional writes depending on whether the record fits directly after setting the flags, since the
    // lowering of the memory accesses below might set flags too
    auto offset = assign(it, TYPE_INT32, "%printf_offset") = used;
    auto stored = assign(it, TYPE_INT32, "%printf_used") = newUsed;
    auto result = method.addNewLocal(TYPE_INT32, "%printf_result");
    assign(it, result) = INT_ZERO;
    auto overflow = assignNop(it) =
        as_signed{newUsed} > as_signed{Value(Literal(static_cast<uint32_t>(PRINTF_BUFFER_SIZE)), TYPE_INT32)};
    // a record not fitting anymore goes into the spill area after the buffer, and is not printed
    assign(it, offset) = (Value(Literal(static_cast<uint32_t>(PRINTF_BUFFER_SIZE)), TYPE_INT32), overflow);
    assign(it, stored) = (used, overflow);
    assign(it, result) = (INT_MINUS_ONE, overflow);
    it.emplace(std::make_unique<MemoryInstruction>(
        MemoryOperation::WRITE, Value(buffer), Value(stored), Value(INT_ONE), false /* already locked */));
    it.nextInBlock();
    it.emplace(std::make_unique<MethodCall>("vc4cl_mutex_unlock"));
    it.nextInBlock();

    // 3. write the record after the counter word
    auto recordOffset = assign(it, TYPE_INT32, "%printf_offset") = offset + Value(Literal(4u), TYPE_INT32);
    // the host maps the format string (and string arguments) back into the global data via its address
    const Value globalData = method.findOrCreateBuiltin(BuiltinLocal::Type::GLOBAL_DATA_ADDRESS)->createReference();
    std::vector<Value> words{Value(Literal(recordSize), TYPE_INT32), globalData, formatWord};
    words.insert(words.end(), argumentWords.begin(), argumentWords.end());
    for(std::size_t i = 0; i < words.size(); ++i)
    {
        const auto& word = words[i];
        auto address =
            method.addNewLocal(method.createPointerType(word.type, AddressSpace::GLOBAL), "%printf_word_address");
        auto wordOffset = assign(it, TYPE_INT32, "%printf_offset") =
            recordOffset + Value(Literal(static_cast<uint32_t>(i * sizeof(uint32_t))), TYPE_INT32);
        assign(it, address) = buffer + wordOffset;
        it.emplace(std::make_unique<MemoryInstruction>(MemoryOperation::WRITE, Value(address), Value(word)));
        it.nextInBlock();
    }

    // 4. the return value: 0 on success, -1 if the record didn't fit
    if(auto out = call->getOutput())
        it.reset(std::make_unique<MoveOperation>(Value(*out), result));
    else
        it.erase();

    removeUnusedPromotions(method, promotedValues);
}
