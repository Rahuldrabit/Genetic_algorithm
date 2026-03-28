#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ga/core/genome.hpp"

namespace ga {
namespace representations {

// Permutation genome: encodes an ordering of integers [0..n-1].
// Suitable for TSP, scheduling, and other sequencing problems.
class PermutationGenome : public ga::IGenome {
public:
    std::vector<int> order;

    PermutationGenome() = default;

    // Construct an identity permutation of length n.
    explicit PermutationGenome(std::size_t n) : order(n) {
        std::iota(order.begin(), order.end(), 0);
    }

    explicit PermutationGenome(std::vector<int> o)
        : order(std::move(o)) {}

    // Build a random permutation of [0..n-1].
    static PermutationGenome random(std::size_t n, std::mt19937& rng) {
        PermutationGenome g(n);
        std::shuffle(g.order.begin(), g.order.end(), rng);
        return g;
    }

    std::unique_ptr<ga::IGenome> clone() const override {
        return std::make_unique<PermutationGenome>(*this);
    }

    std::string encodingName() const override {
        return "permutation";
    }

    std::size_t size() const { return order.size(); }

    // Verify that the genome is a valid permutation of [0..n-1].
    bool isValid() const {
        std::vector<bool> seen(order.size(), false);
        for (int v : order) {
            if (v < 0 || static_cast<std::size_t>(v) >= order.size()) {
                return false;
            }
            if (seen[static_cast<std::size_t>(v)]) {
                return false;
            }
            seen[static_cast<std::size_t>(v)] = true;
        }
        return true;
    }

    // Returns the position index of the given value.
    std::size_t positionOf(int value) const {
        for (std::size_t i = 0; i < order.size(); ++i) {
            if (order[i] == value) {
                return i;
            }
        }
        throw std::out_of_range("PermutationGenome::positionOf: value not found");
    }
};

} // namespace representations
} // namespace ga
