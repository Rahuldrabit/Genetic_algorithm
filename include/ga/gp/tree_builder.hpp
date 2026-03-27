#pragma once

#include <random>
#include <stdexcept>
#include <vector>

#include "ga/gp/node.hpp"

namespace ga {
namespace gp {

class TreeBuilder {
public:
    explicit TreeBuilder(std::vector<Primitive> primitives)
        : primitives_(std::move(primitives)) {}

    std::unique_ptr<Node> grow(std::size_t maxDepth,
                               ValueType targetType,
                               bool stronglyTyped,
                               std::mt19937& rng) const {
        if (maxDepth == 0) {
            return selectPrimitive(targetType, stronglyTyped, true, rng);
        }

        std::uniform_int_distribution<int> coin(0, 1);
        const bool useTerminal = (coin(rng) == 0);
        auto node = selectPrimitive(targetType, stronglyTyped, useTerminal, rng);

        if (!node) {
            throw std::runtime_error("Failed to build GP node");
        }

        if (node->children.empty()) {
            auto primitive = findPrimitive(node->symbol);
            if (primitive && !primitive->isTerminal) {
                for (ValueType argType : primitive->signature.argTypes) {
                    node->children.push_back(grow(maxDepth - 1, argType, stronglyTyped, rng));
                }
            }
        }

        return node;
    }

private:
    const Primitive* findPrimitive(const std::string& name) const {
        for (const auto& p : primitives_) {
            if (p.name == name) {
                return &p;
            }
        }
        return nullptr;
    }

    std::unique_ptr<Node> selectPrimitive(ValueType targetType,
                                          bool stronglyTyped,
                                          bool terminalOnly,
                                          std::mt19937& rng) const {
        std::vector<const Primitive*> candidates;
        for (const auto& p : primitives_) {
            if (terminalOnly && !p.isTerminal) {
                continue;
            }
            if (stronglyTyped && !isCompatible(targetType, p.signature.returnType)) {
                continue;
            }
            candidates.push_back(&p);
        }

        if (candidates.empty() && terminalOnly) {
            return selectPrimitive(targetType, stronglyTyped, false, rng);
        }
        if (candidates.empty()) {
            throw std::runtime_error("No compatible GP primitive available");
        }

        std::uniform_int_distribution<std::size_t> pick(0, candidates.size() - 1);
        const Primitive* p = candidates[pick(rng)];
        return std::make_unique<Node>(p->name, p->signature.returnType);
    }

    std::vector<Primitive> primitives_;
};

} // namespace gp
} // namespace ga
