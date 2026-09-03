#include <iostream>
#include <memory>
#include <vector>

#include "ga/metaheuristics.hpp"

namespace {

double sphere(const std::vector<double>& x) {
    double sum = 0.0;
    for (double value : x) {
        sum += value * value;
    }
    return 1.0 / (1.0 + sum); // The library maximizes fitness.
}

} // namespace

int main() {
    auto fuzzyController = std::make_shared<ga::fuzzy::FuzzyAdaptiveController>();

    ga::Config gaConfig;
    gaConfig.populationSize = 40;
    gaConfig.generations = 50;
    gaConfig.dimension = 5;
    gaConfig.bounds = {-5.0, 5.0};
    gaConfig.seed = 42;

    ga::pso::PsoConfig psoConfig;
    psoConfig.search.populationSize = 40;
    psoConfig.search.iterations = 50;
    psoConfig.search.dimension = 5;
    psoConfig.search.bounds = {-5.0, 5.0};
    psoConfig.search.seed = 43;
    psoConfig.variant = ga::pso::PsoVariant::Constriction;
    psoConfig.controller = fuzzyController;

    ga::aco::AcorConfig acorConfig;
    acorConfig.search.iterations = 40;
    acorConfig.search.dimension = 5;
    acorConfig.search.bounds = {-5.0, 5.0};
    acorConfig.search.seed = 44;
    acorConfig.archiveSize = 40;
    acorConfig.sampleCount = 20;
    acorConfig.controller = fuzzyController;

    ga::hybrid::MetaheuristicPipeline hybrid;
    hybrid.add(std::make_unique<ga::metaheuristics::GeneticAlgorithmAdapter>(gaConfig))
        .add(std::make_unique<ga::pso::ParticleSwarmOptimizer>(psoConfig))
        .add(std::make_unique<ga::aco::ContinuousAntColonyOptimizer>(acorConfig));

    const auto run = hybrid.optimizeDetailed(sphere);
    std::cout << "Best fitness: " << run.combined.bestFitness << '\n';
    std::cout << "Evaluations: " << run.combined.evaluations << '\n';
    for (const auto& stage : run.stages) {
        std::cout << stage.optimizer << ": " << stage.result.bestFitness << '\n';
    }
    return 0;
}
