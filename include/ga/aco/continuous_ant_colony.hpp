#pragma once

#include <memory>

#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace aco {

struct AcorConfig {
    ga::metaheuristics::SearchConfig search;
    std::size_t archiveSize = 50;
    std::size_t sampleCount = 25;
    double locality = 0.5;
    double convergenceSpeed = 0.85;
    std::shared_ptr<const ga::metaheuristics::IAdaptiveController> controller;
};

class ContinuousAntColonyOptimizer final
    : public ga::metaheuristics::IContinuousOptimizer {
public:
    explicit ContinuousAntColonyOptimizer(AcorConfig config = {});

    std::string name() const override { return "ACOR"; }
    ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const ga::metaheuristics::SeedPopulation& seeds = {}) override;

    const AcorConfig& config() const noexcept { return config_; }

private:
    AcorConfig config_;
};

} // namespace aco
} // namespace ga
