#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ga/gp/type_system.hpp"

namespace ga {
namespace gp {

struct Node {
    std::string symbol;
    ValueType returnType = ValueType::Any;
    std::vector<std::unique_ptr<Node>> children;

    Node() = default;
    Node(std::string s, ValueType t) : symbol(std::move(s)), returnType(t) {}

    std::unique_ptr<Node> clone() const {
        auto out = std::make_unique<Node>(symbol, returnType);
        out->children.reserve(children.size());
        for (const auto& child : children) {
            out->children.push_back(child ? child->clone() : nullptr);
        }
        return out;
    }

    std::size_t size() const {
        std::size_t count = 1;
        for (const auto& child : children) {
            if (child) {
                count += child->size();
            }
        }
        return count;
    }
};

} // namespace gp
} // namespace ga
