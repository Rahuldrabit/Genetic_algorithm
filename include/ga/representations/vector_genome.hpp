#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "ga/core/genome.hpp"

namespace ga {
namespace representations {

template <typename T>
class VectorGenome : public ga::IGenome {
public:
    std::vector<T> genes;

    VectorGenome() = default;
    explicit VectorGenome(std::vector<T> g) : genes(std::move(g)) {}

    std::unique_ptr<ga::IGenome> clone() const override {
        return std::make_unique<VectorGenome<T>>(*this);
    }

    std::string encodingName() const override {
        if constexpr (std::is_same<T, double>::value) {
            return "vector<double>";
        }
        if constexpr (std::is_same<T, int>::value) {
            return "vector<int>";
        }
        return "vector";
    }
};

} // namespace representations
} // namespace ga
