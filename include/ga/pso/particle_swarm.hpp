#pragma once

#include <memory>

#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace pso {

enum class PsoVariant {
    GlobalBest,
    LocalBest,
    Constriction,
    BareBones,
    FullyInformed,
    QuantumBehaved,
    Binary
};

struct PsoConfig {
    ga::metaheuristics::SearchConfig search;
    PsoVariant variant = PsoVariant::GlobalBest;
    double inertia = 0.7298;
    double cognitive = 1.49618;
    double social = 1.49618;
    double constriction = 0.7298;
    double velocityClamp = 0.2;
    double binaryVelocityClamp = 4.0;
    std::size_t neighborhoodRadius = 1;
    double quantumBeta = 0.75;
    std::shared_ptr<const ga::metaheuristics::IAdaptiveController> controller;
};

class ParticleSwarmOptimizer final : public ga::metaheuristics::IContinuousOptimizer {
public:
    explicit ParticleSwarmOptimizer(PsoConfig config = {});

    std::string name() const override;
    ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const ga::metaheuristics::SeedPopulation& seeds = {}) override;

    const PsoConfig& config() const noexcept { return config_; }

private:
    PsoConfig config_;
};

} // namespace pso
} // namespace ga
