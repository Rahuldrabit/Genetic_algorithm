#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace hybrid {

struct StageResult {
    std::string optimizer;
    ga::core::OptimizationResult result;
};

struct PipelineResult {
    ga::core::OptimizationResult combined;
    std::vector<StageResult> stages;
};

// Sequential heterogeneous hybrid. Each stage receives the best solutions
// found by prior stages as seeds, so this is solution transfer rather than a
// simple winner-takes-all portfolio.
class MetaheuristicPipeline final
    : public ga::metaheuristics::IContinuousOptimizer {
public:
    MetaheuristicPipeline& add(
        std::unique_ptr<ga::metaheuristics::IContinuousOptimizer> optimizer);

    std::string name() const override { return "hybrid-pipeline"; }
    ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const ga::metaheuristics::SeedPopulation& seeds = {}) override;
    PipelineResult optimizeDetailed(
        const ga::Fitness& fitness,
        const ga::metaheuristics::SeedPopulation& seeds = {});

    std::size_t size() const noexcept { return stages_.size(); }

private:
    std::vector<std::unique_ptr<ga::metaheuristics::IContinuousOptimizer>> stages_;
};

} // namespace hybrid
} // namespace ga
