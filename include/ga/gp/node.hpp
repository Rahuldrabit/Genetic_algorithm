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

    Node(const Node& other) : symbol(other.symbol), returnType(other.returnType) {
        children.reserve(other.children.size());
        for (const auto& child : other.children) {
            children.push_back(child ? child->clone() : nullptr);
        }
    }

    Node& operator=(const Node& other) {
        if (this == &other) {
            return *this;
        }
        symbol = other.symbol;
        returnType = other.returnType;
        children.clear();
        children.reserve(other.children.size());
        for (const auto& child : other.children) {
            children.push_back(child ? child->clone() : nullptr);
        }
        return *this;
    }

    Node(Node&&) noexcept = default;
    Node& operator=(Node&&) noexcept = default;

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
