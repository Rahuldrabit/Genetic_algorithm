#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ga/core/genome.hpp"

namespace ga {
namespace representations {

// Fixed-length binary genome backed by a vector<bool>.
// Suitable for combinatorial problems, feature selection, and binary encoding.
class BitsetGenome : public ga::IGenome {
public:
    std::vector<bool> bits;

    BitsetGenome() = default;
    explicit BitsetGenome(std::size_t size, bool fill = false)
        : bits(size, fill) {}
    explicit BitsetGenome(std::vector<bool> b)
        : bits(std::move(b)) {}

    std::unique_ptr<ga::IGenome> clone() const override {
        return std::make_unique<BitsetGenome>(*this);
    }

    std::string encodingName() const override {
        return "bitset";
    }

    std::size_t size() const { return bits.size(); }

    // Hamming distance to another BitsetGenome.
    std::size_t hammingDistance(const BitsetGenome& other) const {
        std::size_t dist = 0;
        std::size_t len = std::min(bits.size(), other.bits.size());
        for (std::size_t i = 0; i < len; ++i) {
            if (bits[i] != other.bits[i]) {
                ++dist;
            }
        }
        dist += (bits.size() > len ? bits.size() - len : 0);
        dist += (other.bits.size() > len ? other.bits.size() - len : 0);
        return dist;
    }

    // Count number of set bits (ones).
    std::size_t popcount() const {
        std::size_t count = 0;
        for (bool b : bits) {
            if (b) ++count;
        }
        return count;
    }
};

} // namespace representations
} // namespace ga
