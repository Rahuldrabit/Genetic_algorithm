#pragma once

#include <cstddef>
#include "ga/config.hpp"

namespace ga {
namespace shake {

enum class ShakeType {
    Uniform,
    Gaussian,
    Cauchy,
    LevyFlight,
    Opposition,
    PartialDimension
};

struct ShakeConfig {
    ShakeType type = ShakeType::Cauchy;
    double intensity = 0.1;        // Scale relative to domain span
    double probability = 1.0;      // Probability of applying shake to an individual
    double levyBeta = 1.5;         // Exponent for Levy flight [1.0, 2.0]
    std::size_t partialDims = 2;   // Dimensions to mutate in PartialDimension shake
    bool retainBest = true;        // Keep original if shake produces worse fitness
};

} // namespace shake
} // namespace ga
