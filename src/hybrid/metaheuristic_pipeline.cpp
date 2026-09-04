#include "ga/hybrid/metaheuristic_pipeline.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ga {
namespace hybrid {

MetaheuristicPipeline& MetaheuristicPipeline::add(
    std::unique_ptr<ga::metaheuristics::IContinuousOptimizer> optimizer) {
    return addShared(std::shared_ptr<ga::metaheuristics::IContinuousOptimizer>(
        std::move(optimizer)));
}

MetaheuristicPipeline& MetaheuristicPipeline::addShared(
    std::shared_ptr<ga::metaheuristics::IContinuousOptimizer> optimizer) {
    if (!optimizer) {
        throw std::invalid_argument("cannot add a null optimizer to the hybrid pipeline");
    }
    stages_.push_back(std::move(optimizer));
    return *this;
}

ga::core::OptimizationResult MetaheuristicPipeline::optimize(
    const ga::Fitness& fitness,
    const ga::metaheuristics::SeedPopulation& seeds) {
    return optimizeDetailed(fitness, seeds).combined;
}

PipelineResult MetaheuristicPipeline::optimizeDetailed(
    const ga::Fitness& fitness,
    const ga::metaheuristics::SeedPopulation& seeds) {
    if (!fitness) {
        throw std::invalid_argument("fitness callback is empty");
    }
    if (stages_.empty()) {
        throw std::logic_error("hybrid pipeline contains no optimizers");
    }

    PipelineResult pipeline;
    pipeline.combined.bestFitness = -std::numeric_limits<double>::infinity();
    ga::metaheuristics::SeedPopulation transfer = seeds;
    pipeline.stages.reserve(stages_.size());

    for (auto& optimizer : stages_) {
        StageResult stage;
        stage.optimizer = optimizer->name();
        stage.result = optimizer->optimize(fitness, transfer);
        if (stage.result.bestSolution.empty()) {
            throw std::runtime_error(stage.optimizer + " returned an empty best solution");
        }

        if (pipeline.combined.bestSolution.empty() ||
            stage.result.bestFitness > pipeline.combined.bestFitness) {
            pipeline.combined.bestSolution = stage.result.bestSolution;
            pipeline.combined.bestFitness = stage.result.bestFitness;
        }
        pipeline.combined.bestHistory.insert(
            pipeline.combined.bestHistory.end(),
            stage.result.bestHistory.begin(),
            stage.result.bestHistory.end());
        pipeline.combined.avgHistory.insert(
            pipeline.combined.avgHistory.end(),
            stage.result.avgHistory.begin(),
            stage.result.avgHistory.end());
        pipeline.combined.evaluations += stage.result.evaluations;
        pipeline.combined.generations += stage.result.generations;

        transfer.clear();
        transfer.push_back(stage.result.bestSolution);
        if (stage.result.bestSolution != pipeline.combined.bestSolution) {
            transfer.push_back(pipeline.combined.bestSolution);
        }
        pipeline.stages.push_back(std::move(stage));
    }
    return pipeline;
}

} // namespace hybrid
} // namespace ga
