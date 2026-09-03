#pragma once

#include <utility>

#include "ga/genetic_algorithm.hpp"
#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace metaheuristics {

class GeneticAlgorithmAdapter final : public IContinuousOptimizer {
public:
    explicit GeneticAlgorithmAdapter(ga::Config config)
        : config_(std::move(config)) {}

    std::string name() const override { return "GA"; }

    ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const SeedPopulation& seeds = {}) override {
        ga::GeneticAlgorithm algorithm(config_);
        const ga::Result source = algorithm.run(fitness, seeds);

        ga::core::OptimizationResult result;
        result.bestSolution = source.bestGenes;
        result.bestFitness = source.bestFitness;
        result.bestHistory = source.bestHistory;
        result.avgHistory = source.avgHistory;
        result.evaluations = source.evaluations;
        result.generations = source.iterations;
        return result;
    }

private:
    ga::Config config_;
};

} // namespace metaheuristics
} // namespace ga
