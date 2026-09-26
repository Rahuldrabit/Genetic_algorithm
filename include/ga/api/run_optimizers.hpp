#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ga/config.hpp"
#include "ga/core/result.hpp"
#include "ga/aco/ant_colony.hpp"
#include "ga/aco/continuous_ant_colony.hpp"
#include "ga/pso/particle_swarm.hpp"
#include "ga/pso/clpso.hpp"
#include "ga/gso/glowworm_swarm.hpp"
#include "ga/gsa/gravitational_search.hpp"
#include "ga/metaheuristics/genetic_algorithm_adapter.hpp"
#include "ga/algorithms/island_model.hpp"
#include "ga/metaheuristics/simulated_annealing.hpp"
#include "ga/metaheuristics/hill_climbing.hpp"
#include "ga/adaptive/dynamic_ga.hpp"
#include "ga/constraints/constrained_optimizer.hpp"

namespace ga {
namespace api {

enum class AlgorithmType {
    GlobalBestPso,
    LocalBestPso,
    ConstrictionPso,
    BareBonesPso,
    FullyInformedPso,
    QuantumBehavedPso,
    BinaryPso,
    ComprehensiveLearningPso,

    AcorGaussian,

    GsoStandard,
    GsoAdaptiveStep,
    GsoLevyFlight,
    GsoMultiModal,

    GravitationalSearch,
    GeneticAlgorithm,

    // Alternatives & Advanced Strategies
    IslandModel,
    SimulatedAnnealing,
    HillClimbing,
    DynamicGeneticAlgorithm,
    ConstrainedOptimization
};

namespace detail {
inline ga::metaheuristics::SearchConfig makeSearchConfig(
    std::size_t dimension,
    std::size_t iterations,
    ga::Bounds bounds,
    unsigned seed,
    std::size_t populationSize) {
    ga::metaheuristics::SearchConfig sc;
    sc.dimension = dimension;
    sc.iterations = iterations;
    sc.populationSize = populationSize;
    sc.seed = seed;
    sc.bounds = bounds;
    return sc;
}
} // namespace detail

inline ga::core::OptimizationResult runPso(
    const ga::Fitness& fitness,
    ga::pso::PsoVariant variant = ga::pso::PsoVariant::GlobalBest,
    std::size_t dimension = 10,
    std::size_t iterations = 100,
    ga::Bounds bounds = {-5.12, 5.12},
    unsigned seed = 0,
    std::size_t populationSize = 50) {
    ga::pso::PsoConfig cfg;
    cfg.search = detail::makeSearchConfig(dimension, iterations, bounds, seed, populationSize);
    cfg.variant = variant;
    ga::pso::ParticleSwarmOptimizer opt(cfg);
    return opt.optimize(fitness);
}

inline ga::core::OptimizationResult runClpso(
    const ga::Fitness& fitness,
    std::size_t dimension = 10,
    std::size_t iterations = 100,
    ga::Bounds bounds = {-5.12, 5.12},
    unsigned seed = 0,
    std::size_t populationSize = 50) {
    ga::pso::ClpsoConfig cfg;
    cfg.search = detail::makeSearchConfig(dimension, iterations, bounds, seed, populationSize);
    ga::pso::ComprehensiveLearningPso opt(cfg);
    return opt.optimize(fitness);
}

inline ga::core::OptimizationResult runContinuousAco(
    const ga::Fitness& fitness,
    std::size_t dimension = 10,
    std::size_t iterations = 100,
    ga::Bounds bounds = {-5.12, 5.12},
    unsigned seed = 0,
    std::size_t archiveSize = 50) {
    ga::aco::AcorConfig cfg;
    cfg.search = detail::makeSearchConfig(dimension, iterations, bounds, seed, archiveSize);
    cfg.archiveSize = archiveSize;
    ga::aco::ContinuousAntColonyOptimizer opt(cfg);
    return opt.optimize(fitness);
}

inline ga::aco::AntColonyResult runGraphAco(
    const ga::aco::DenseGraph& graph,
    ga::aco::AntColonyVariant variant = ga::aco::AntColonyVariant::AntSystem,
    std::size_t ants = 30,
    std::size_t iterations = 100,
    unsigned seed = 0) {
    ga::aco::AntColonyConfig cfg;
    cfg.variant = variant;
    cfg.ants = ants;
    cfg.iterations = iterations;
    cfg.seed = seed;
    ga::aco::AntColonyOptimizer opt(cfg);
    return opt.solve(graph);
}

inline ga::core::OptimizationResult runGso(
    const ga::Fitness& fitness,
    ga::gso::GsoVariant variant = ga::gso::GsoVariant::Standard,
    std::size_t dimension = 10,
    std::size_t iterations = 100,
    ga::Bounds bounds = {-5.12, 5.12},
    unsigned seed = 0,
    std::size_t populationSize = 50) {
    ga::gso::GsoConfig cfg;
    cfg.search = detail::makeSearchConfig(dimension, iterations, bounds, seed, populationSize);
    cfg.variant = variant;
    ga::gso::GlowwormSwarmOptimizer opt(cfg);
    return opt.optimize(fitness);
}

inline ga::core::OptimizationResult runGsa(
    const ga::Fitness& fitness,
    std::size_t dimension = 10,
    std::size_t iterations = 100,
    ga::Bounds bounds = {-5.12, 5.12},
    unsigned seed = 0,
    std::size_t populationSize = 50) {
    ga::gsa::GsaConfig cfg;
    cfg.search = detail::makeSearchConfig(dimension, iterations, bounds, seed, populationSize);
    ga::gsa::GravitationalSearchOptimizer opt(cfg);
    return opt.optimize(fitness);
}

inline ga::core::OptimizationResult runGeneticAlgorithm(
    const ga::Fitness& fitness,
    std::size_t dimension = 10,
    std::size_t generations = 100,
    ga::Bounds bounds = {-5.12, 5.12},
    unsigned seed = 0,
    std::size_t populationSize = 50) {
    ga::Config cfg;
    cfg.populationSize = static_cast<int>(populationSize);
    cfg.generations = static_cast<int>(generations);
    cfg.dimension = static_cast<int>(dimension);
    cfg.seed = seed;
    cfg.bounds = bounds;
    ga::metaheuristics::GeneticAlgorithmAdapter opt(cfg);
    return opt.optimize(fitness);
}

// ---------------- High-Level Runners for Alternatives ----------------

inline ga::core::OptimizationResult runIslandModel(
    const ga::Fitness& fitness,
    std::size_t numIslands = 4,
    std::size_t islandPopulationSize = 50,
    std::size_t dimension = 10,
    std::size_t iterations = 100,
    ga::Bounds bounds = {-5.12, 5.12},
    unsigned seed = 0,
    ga::algorithms::IslandTopology topology = ga::algorithms::IslandTopology::Ring) {
    ga::algorithms::IslandConfig cfg;
    cfg.numIslands = numIslands;
    cfg.islandPopulationSize = islandPopulationSize;
    cfg.iterations = iterations;
    cfg.topology = topology;
    cfg.search = detail::makeSearchConfig(dimension, iterations, bounds, seed, numIslands * islandPopulationSize);
    ga::algorithms::IslandModelOptimizer opt(cfg);
    return opt.optimize(fitness);
}

inline ga::core::OptimizationResult runSimulatedAnnealing(
    const ga::Fitness& fitness,
    std::size_t dimension = 10,
    std::size_t iterations = 200,
    ga::Bounds bounds = {-5.12, 5.12},
    double initialTemperature = 100.0,
    double coolingRate = 0.95,
    unsigned seed = 0) {
    ga::metaheuristics::SimulatedAnnealingConfig cfg;
    cfg.search = detail::makeSearchConfig(dimension, iterations, bounds, seed, 1);
    cfg.initialTemperature = initialTemperature;
    cfg.coolingRate = coolingRate;
    ga::metaheuristics::SimulatedAnnealingOptimizer opt(cfg);
    return opt.optimize(fitness);
}

inline ga::core::OptimizationResult runHillClimbing(
    const ga::Fitness& fitness,
    std::size_t dimension = 10,
    std::size_t iterations = 200,
    ga::Bounds bounds = {-5.12, 5.12},
    double stepSize = 0.02,
    std::size_t restarts = 2,
    unsigned seed = 0) {
    ga::metaheuristics::HillClimbingConfig cfg;
    cfg.search = detail::makeSearchConfig(dimension, iterations, bounds, seed, 1);
    cfg.stepSize = stepSize;
    cfg.restarts = restarts;
    ga::metaheuristics::HillClimbingOptimizer opt(cfg);
    return opt.optimize(fitness);
}

inline ga::core::OptimizationResult runDynamicGa(
    const ga::Fitness& fitness,
    std::size_t dimension = 10,
    std::size_t iterations = 100,
    ga::Bounds bounds = {-5.12, 5.12},
    std::size_t populationSize = 50,
    unsigned seed = 0,
    ga::adaptive::DynamicStrategy strategy = ga::adaptive::DynamicStrategy::Hybrid) {
    ga::adaptive::DynamicGAConfig cfg;
    cfg.search = detail::makeSearchConfig(dimension, iterations, bounds, seed, populationSize);
    cfg.strategy = strategy;
    ga::adaptive::DynamicGAOptimizer opt(cfg);
    return opt.optimize(fitness);
}

inline ga::core::OptimizationResult runConstrainedOptimization(
    const ga::Fitness& fitness,
    const ga::constraints::ConstraintSet& constraints,
    std::size_t dimension = 10,
    std::size_t iterations = 100,
    ga::Bounds bounds = {-5.12, 5.12},
    std::size_t populationSize = 50,
    unsigned seed = 0) {
    ga::constraints::ConstrainedOptimizerConfig cfg;
    cfg.search = detail::makeSearchConfig(dimension, iterations, bounds, seed, populationSize);
    ga::constraints::ConstrainedOptimizer opt(cfg, constraints);
    return opt.optimize(fitness);
}

inline ga::core::OptimizationResult solve(
    AlgorithmType type,
    const ga::Fitness& fitness,
    const ga::metaheuristics::SearchConfig& config = {}) {
    const std::size_t dim = config.dimension;
    const std::size_t iters = config.iterations;
    const ga::Bounds b = config.bounds;
    const unsigned seed = config.seed;
    const std::size_t pop = config.populationSize;

    switch (type) {
        case AlgorithmType::GlobalBestPso:
            return runPso(fitness, ga::pso::PsoVariant::GlobalBest, dim, iters, b, seed, pop);
        case AlgorithmType::LocalBestPso:
            return runPso(fitness, ga::pso::PsoVariant::LocalBest, dim, iters, b, seed, pop);
        case AlgorithmType::ConstrictionPso:
            return runPso(fitness, ga::pso::PsoVariant::Constriction, dim, iters, b, seed, pop);
        case AlgorithmType::BareBonesPso:
            return runPso(fitness, ga::pso::PsoVariant::BareBones, dim, iters, b, seed, pop);
        case AlgorithmType::FullyInformedPso:
            return runPso(fitness, ga::pso::PsoVariant::FullyInformed, dim, iters, b, seed, pop);
        case AlgorithmType::QuantumBehavedPso:
            return runPso(fitness, ga::pso::PsoVariant::QuantumBehaved, dim, iters, b, seed, pop);
        case AlgorithmType::BinaryPso:
            return runPso(fitness, ga::pso::PsoVariant::Binary, dim, iters, b, seed, pop);
        case AlgorithmType::ComprehensiveLearningPso:
            return runClpso(fitness, dim, iters, b, seed, pop);
        case AlgorithmType::AcorGaussian:
            return runContinuousAco(fitness, dim, iters, b, seed, pop);
        case AlgorithmType::GsoStandard:
            return runGso(fitness, ga::gso::GsoVariant::Standard, dim, iters, b, seed, pop);
        case AlgorithmType::GsoAdaptiveStep:
            return runGso(fitness, ga::gso::GsoVariant::AdaptiveStep, dim, iters, b, seed, pop);
        case AlgorithmType::GsoLevyFlight:
            return runGso(fitness, ga::gso::GsoVariant::LevyFlight, dim, iters, b, seed, pop);
        case AlgorithmType::GsoMultiModal:
            return runGso(fitness, ga::gso::GsoVariant::MultiModal, dim, iters, b, seed, pop);
        case AlgorithmType::GravitationalSearch:
            return runGsa(fitness, dim, iters, b, seed, pop);
        case AlgorithmType::GeneticAlgorithm:
            return runGeneticAlgorithm(fitness, dim, iters, b, seed, pop);
        case AlgorithmType::IslandModel:
            return runIslandModel(fitness, 4, std::max<std::size_t>(10, pop / 4), dim, iters, b, seed);
        case AlgorithmType::SimulatedAnnealing:
            return runSimulatedAnnealing(fitness, dim, iters, b, 100.0, 0.95, seed);
        case AlgorithmType::HillClimbing:
            return runHillClimbing(fitness, dim, iters, b, 0.02, 2, seed);
        case AlgorithmType::DynamicGeneticAlgorithm:
            return runDynamicGa(fitness, dim, iters, b, pop, seed);
        case AlgorithmType::ConstrainedOptimization:
            return runConstrainedOptimization(fitness, {}, dim, iters, b, pop, seed);
        default:
            return runPso(fitness, ga::pso::PsoVariant::GlobalBest, dim, iters, b, seed, pop);
    }
}

} // namespace api
} // namespace ga
