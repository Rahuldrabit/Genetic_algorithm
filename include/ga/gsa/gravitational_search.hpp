#pragma once

#include <memory>

#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace gsa {

struct GsaConfig {
    ga::metaheuristics::SearchConfig search;
    double gravitationalConstant = 100.0;
    double decay = 20.0;
    double epsilon = 1e-12;
    double finalEliteFraction = 0.02;
    std::shared_ptr<const ga::metaheuristics::IAdaptiveController> controller;
};

class GravitationalSearchOptimizer final
    : public ga::metaheuristics::IContinuousOptimizer {
public:
    explicit GravitationalSearchOptimizer(GsaConfig config = {});

    std::string name() const override { return "GSA"; }
    ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const ga::metaheuristics::SeedPopulation& seeds = {}) override;

    const GsaConfig& config() const noexcept { return config_; }

private:
    GsaConfig config_;
};

} // namespace gsa
} // namespace ga
