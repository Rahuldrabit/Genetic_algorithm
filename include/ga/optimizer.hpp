#pragma once
/// High-Level API – optimizer.hpp
///
/// Wraps the complete GA pipeline behind a single, friendly interface.
///
/// Usage (minimal):
///   ga::optimizer::Problem prob;
///   prob.dimension = 10;
///   prob.bounds    = {-5.12, 5.12};
///   prob.fitness   = [](const std::vector<double>& x){ return -sphere(x); };
///
///   ga::optimizer::Optimizer opt;
///   auto result = opt.optimize(prob);
///   std::cout << result.bestFitness << '\n';
///
/// Usage (full control):
///   ga::optimizer::Optimizer opt;
///   opt.cfg.populationSize = 100;
///   opt.cfg.generations    = 200;
///   opt.logger             = std::make_shared<ga::experiment::ExperimentLogger>();
///   opt.adaptiveRates      = true;
///   auto result = opt.optimize(prob);

#include "ga/config.hpp"
#include "ga/genetic_algorithm.hpp"
#include "ga/adaptive_operators.hpp"
#include "ga/constraint.hpp"
#include "ga/experiment.hpp"
#include "ga/visualization.hpp"
#include "ga/parallel_eval.hpp"

#include <memory>
#include <vector>
#include <functional>

namespace ga {
namespace optimizer {

// ============================================================================
// Problem – everything that defines an optimisation task
// ============================================================================

struct Problem {
    int    dimension     = 10;
    Bounds bounds        = {-5.12, 5.12};
    Fitness fitness;                          ///< required: genes → double

    /// Optional constraints (empty = unconstrained).
    std::vector<constraint::Constraint> constraints;
    double penaltyWeight = 1000.0;            ///< weight for penalty method
};

// ============================================================================
// OptimizerResult
// ============================================================================

struct OptimizerResult {
    std::vector<double> bestGenes;
    double bestFitness = -1e300;
    std::vector<double> bestHistory;
    std::vector<double> avgHistory;
    std::vector<double> diversityHistory;
};

// ============================================================================
// Optimizer – high-level one-call interface
// ============================================================================

class Optimizer {
public:
    Config cfg;                              ///< standard GA configuration
    bool   adaptiveRates  = false;           ///< enable adaptive mutation/crossover
    bool   parallelEval   = false;           ///< evaluate fitness in parallel
    int    numThreads     = 0;               ///< 0 = hardware_concurrency
    std::shared_ptr<experiment::ExperimentLogger> logger;  ///< optional logger

    OptimizerResult optimize(const Problem& prob);

private:
    struct Individual {
        std::vector<double> genes;
        double fitness = 0.0;
    };

    std::mt19937 rng_;

    std::vector<Individual> initPop(const Problem& prob, int popSize);
    std::pair<Individual,Individual> crossoverPair(const Individual& a,
                                                    const Individual& b,
                                                    const Problem& prob);
    void mutate(Individual& ind, double mutRate, const Problem& prob);
    void clip(std::vector<double>& g, const Problem& prob) const;
    const Individual& tournamentSelect(const std::vector<Individual>& pop);
    double evalFitness(const std::vector<double>& genes, const Problem& prob) const;
};

} // namespace optimizer
} // namespace ga
