#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace gso {

enum class GsoVariant {
    Standard,
    AdaptiveStep,
    LevyFlight,
    MultiModal
};

struct GsoConfig {
    ga::metaheuristics::SearchConfig search;
    GsoVariant variant = GsoVariant::Standard;

    double rho = 0.4;                  // Luciferin decay constant in (0, 1)
    double gamma = 0.6;                // Luciferin enhancement fraction
    double beta = 0.08;                // Decision range gain
    double stepSize = 0.03;            // Step size as fraction of bound span
    double minStepSize = 0.005;        // Minimum step for AdaptiveStep
    double maxStepSize = 0.08;         // Maximum step for AdaptiveStep
    double sensorRange = 1.0;          // Max sensor radius as fraction of span
    double initialSensorRange = 0.25;  // Initial sensor radius fraction
    std::size_t targetNeighbors = 5;   // Desired neighborhood size (nt)
    double initialLuciferin = 5.0;     // Initial luciferin level
    double nicheRadius = 0.1;          // Minimum distance between peaks in MultiModal

    std::shared_ptr<const ga::metaheuristics::IAdaptiveController> controller;
};

struct Glowworm {
    std::vector<double> position;
    double luciferin = 5.0;
    double sensorRadius = 0.25;
    double fitness = -1e300;
};

} // namespace gso
} // namespace ga
