/*
 * Author: doe300
 *
 * See the file "LICENSE" for the full license governing this code.
 */

#include "FloatLiterals.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace vc4c;

static bool isIdentifierCharacter(char c)
{
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

/*
 * Returns the end of the preprocessing number starting at pos (C99 6.4.8):
 * digit or . digit, followed by digits, identifier characters, e+ e- E+ E- p+ p- P+ P- and dots.
 */
static std::size_t findEndOfNumber(const std::string& source, std::size_t pos)
{
    ++pos;
    while(pos < source.size())
    {
        char c = source[pos];
        if((c == '+' || c == '-') && std::strchr("eEpP", source[pos - 1]))
            ++pos;
        else if(isIdentifierCharacter(c) || c == '.')
            ++pos;
        else
            break;
    }
    return pos;
}

/*
 * Checks whether the preprocessing number is a floating-point literal with an optional f/F suffix and splits it into
 * the value and the suffix. Hexadecimal literals are included, since they can also have more digits than fit.
 */
static bool splitFloatLiteral(const std::string& number, std::string& value, std::string& suffix)
{
    std::size_t pos = 0;
    const bool isHex = number.size() > 2 && number[0] == '0' && (number[1] == 'x' || number[1] == 'X');
    auto isDigit = [isHex](char c) {
        return isHex ? std::isxdigit(static_cast<unsigned char>(c)) : std::isdigit(static_cast<unsigned char>(c));
    };
    if(isHex)
        pos = 2;
    std::size_t numDigits = 0;
    while(pos < number.size() && isDigit(number[pos]))
    {
        ++pos;
        ++numDigits;
    }
    bool hasDot = false;
    if(pos < number.size() && number[pos] == '.')
    {
        hasDot = true;
        ++pos;
        while(pos < number.size() && isDigit(number[pos]))
        {
            ++pos;
            ++numDigits;
        }
    }
    if(numDigits == 0)
        return false;
    bool hasExponent = false;
    if(pos < number.size() && std::strchr(isHex ? "pP" : "eE", number[pos]))
    {
        hasExponent = true;
        ++pos;
        if(pos < number.size() && (number[pos] == '+' || number[pos] == '-'))
            ++pos;
        std::size_t numExponentDigits = 0;
        while(pos < number.size() && std::isdigit(static_cast<unsigned char>(number[pos])))
        {
            ++pos;
            ++numExponentDigits;
        }
        if(numExponentDigits == 0)
            return false;
    }
    // hexadecimal floating-point literals require the exponent, decimal ones a dot or an exponent
    if(isHex ? !hasExponent : !(hasDot || hasExponent))
        return false;
    value = number.substr(0, pos);
    suffix = number.substr(pos);
    // half (h), long double (l) and other suffixes are not handled
    return suffix.empty() || suffix == "f" || suffix == "F";
}

/*
 * Returns the hexadecimal literal of the value of the literal rounded toward zero, if it differs from the value
 * rounded to nearest and is a finite normal value.
 */
static bool convertTowardZero(const std::string& value, bool isDouble, std::string& hexLiteral)
{
    const int previousRounding = std::fegetround();
    std::fesetround(FE_TONEAREST);
    errno = 0;
    // the value clang uses: a double literal is converted to double and then to float
    const float nearest =
        isDouble ? static_cast<float>(std::strtod(value.data(), nullptr)) : std::strtof(value.data(), nullptr);
    const bool nearestInRange = errno == 0;
    std::fesetround(FE_TOWARDZERO);
    errno = 0;
    const float towardZero = std::strtof(value.data(), nullptr);
    const bool towardZeroInRange = errno == 0;
    std::fesetround(previousRounding);

    if(!nearestInRange || !towardZeroInRange || nearest == towardZero)
        return false;
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%a", static_cast<double>(towardZero));
    hexLiteral = buffer;
    return true;
}

/*
 * Returns whether the preprocessor directive starting at pos (the '#') is one whose operands are not C tokens.
 */
static bool isSkippedDirective(const std::string& source, std::size_t pos)
{
    ++pos;
    while(pos < source.size() && (source[pos] == ' ' || source[pos] == '\t'))
        ++pos;
    for(const char* directive : {"include", "pragma", "line", "error", "warning"})
    {
        auto length = std::strlen(directive);
        if(source.compare(pos, length, directive) == 0 &&
            (pos + length == source.size() || !isIdentifierCharacter(source[pos + length])))
            return true;
    }
    return false;
}

// Returns the position after the end of the line, honoring line continuations
static std::size_t skipLine(const std::string& source, std::size_t pos)
{
    while(pos < source.size() && source[pos] != '\n')
    {
        if(source[pos] == '\\' && pos + 1 < source.size() && source[pos + 1] == '\n')
            ++pos;
        ++pos;
    }
    return pos;
}

bool precompilation::roundFloatLiteralsTowardZero(const std::string& source, std::string& result)
{
    result.clear();
    result.reserve(source.size());
    bool changed = false;
    // whether only whitespace precedes the current position in the (logical) line
    bool atLineStart = true;
    std::size_t pos = 0;
    while(pos < source.size())
    {
        const char c = source[pos];
        const char next = pos + 1 < source.size() ? source[pos + 1] : '\0';
        std::size_t end = pos + 1;
        if(c == '/' && next == '/')
            end = skipLine(source, pos);
        else if(c == '/' && next == '*')
        {
            end = source.find("*/", pos + 2);
            end = end == std::string::npos ? source.size() : end + 2;
        }
        else if(c == '"' || c == '\'')
        {
            // string or character literal (also the prefixed ones, the prefix is an identifier before)
            while(end < source.size() && source[end] != c && source[end] != '\n')
                end += source[end] == '\\' ? 2 : 1;
            end = std::min(end + 1, source.size());
        }
        else if(c == '#' && atLineStart && isSkippedDirective(source, pos))
            end = skipLine(source, pos);
        else if(isIdentifierCharacter(c) && !std::isdigit(static_cast<unsigned char>(c)))
        {
            // identifier (or keyword), may contain digits
            while(end < source.size() && isIdentifierCharacter(source[end]))
                ++end;
        }
        else if(std::isdigit(static_cast<unsigned char>(c)) ||
            (c == '.' && std::isdigit(static_cast<unsigned char>(next))))
        {
            end = findEndOfNumber(source, pos);
            const std::string number = source.substr(pos, end - pos);
            std::string value, suffix, hexLiteral;
            if(splitFloatLiteral(number, value, suffix) && convertTowardZero(value, suffix.empty(), hexLiteral))
            {
                result.append(hexLiteral).append(suffix);
                changed = true;
                atLineStart = false;
                pos = end;
                continue;
            }
        }
        if(c == '\n')
            atLineStart = true;
        else if(!std::isspace(static_cast<unsigned char>(c)) && !(c == '/' && (next == '*' || next == '/')))
            atLineStart = false;
        result.append(source, pos, end - pos);
        pos = end;
    }
    return changed;
}
