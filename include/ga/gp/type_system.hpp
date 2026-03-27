#pragma once

#include <string>
#include <vector>

namespace ga {
namespace gp {

enum class ValueType {
    Any,
    Bool,
    Int,
    Double
};

inline bool isCompatible(ValueType expected, ValueType actual) {
    return expected == ValueType::Any || actual == ValueType::Any || expected == actual;
}

struct Signature {
    ValueType returnType = ValueType::Any;
    std::vector<ValueType> argTypes;
};

struct Primitive {
    std::string name;
    Signature signature;
    bool isTerminal = false;
};

} // namespace gp
} // namespace ga
