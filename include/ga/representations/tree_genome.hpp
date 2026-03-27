#pragma once

#include <memory>
#include <string>

#include "ga/core/genome.hpp"
#include "ga/gp/node.hpp"

namespace ga {
namespace representations {

class TreeGenome : public ga::IGenome {
public:
    std::unique_ptr<ga::gp::Node> root;

    TreeGenome() = default;
    explicit TreeGenome(std::unique_ptr<ga::gp::Node> r) : root(std::move(r)) {}

    TreeGenome(const TreeGenome& other) {
        root = other.root ? other.root->clone() : nullptr;
    }

    TreeGenome& operator=(const TreeGenome& other) {
        if (this == &other) {
            return *this;
        }
        root = other.root ? other.root->clone() : nullptr;
        return *this;
    }

    TreeGenome(TreeGenome&&) noexcept = default;
    TreeGenome& operator=(TreeGenome&&) noexcept = default;

    std::unique_ptr<ga::IGenome> clone() const override {
        return std::make_unique<TreeGenome>(*this);
    }

    std::string encodingName() const override {
        return "tree";
    }
};

} // namespace representations
} // namespace ga
