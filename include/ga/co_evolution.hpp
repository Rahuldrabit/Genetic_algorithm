#pragma once
/// Co-Evolution (Multi-Population)
///
/// Two flavours:
///   CooperativeCoEvolution  – sub-populations each evolve a component; fitness
///                             is evaluated by combining representatives from
///                             all populations.
///   CompetitiveCoEvolution  – two antagonistic populations (e.g. predator /
///                             prey). Each individual's fitness depends on how
///                             well it performs against the other population.

#include <vector>
#include <functional>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ga {
namespace coev {

// ============================================================================
// CoEvConfig
// ============================================================================

struct CoEvConfig {
    int    numPopulations  = 2;   ///< number of co-evolving populations
    int    popSize         = 50;  ///< individuals per population
    int    generations     = 100;
    int    dimension       = 5;   ///< genes per individual per sub-population
    double crossoverRate   = 0.8;
    double mutationRate    = 0.1;
    double lowerBound      = -5.0;
    double upperBound      =  5.0;
    unsigned seed          = 0;
};

// ============================================================================
// CoEvResult
// ============================================================================

struct CoEvResult {
    /// Best representative from each population (indices match populations).
    std::vector<std::vector<double>> bestPerPop;
    std::vector<double>              bestFitnessPerPop;
    std::vector<double>              bestHistory;  ///< overall best per generation
};

// ============================================================================
// CooperativeCoEvolution
///
/// Each population evolves one component of the solution vector.
/// A complete solution is assembled by concatenating one individual from each
/// population.  The combined fitness function receives the full concatenated
/// vector and returns a scalar.
// ============================================================================

class CooperativeCoEvolution {
public:
    /// @p fitness  Receives the full concatenated gene vector and returns fitness.
    using FitnessFunc = std::function<double(const std::vector<double>&)>;

    explicit CooperativeCoEvolution(const CoEvConfig& cfg) : cfg_(cfg) {
        unsigned seed = cfg_.seed == 0 ? std::random_device{}() : cfg_.seed;
        rng_.seed(seed);
    }

    CoEvResult run(const FitnessFunc& fitness);

private:
    struct Individual { std::vector<double> genes; double fit = -1e300; };
    using Pop = std::vector<Individual>;

    CoEvConfig   cfg_;
    std::mt19937 rng_;

    Pop initPop(int dim);
    Individual crossover(const Individual& a, const Individual& b);
    void mutate(Individual& ind);
    void clip(std::vector<double>& x) const;
    const Individual& tournament(const Pop& pop);

    /// Build full gene vector by taking ind from pop p and best reps elsewhere.
    std::vector<double> assemble(int popIdx,
                                 const Individual& ind,
                                 const std::vector<Individual>& reps) const;
};

// ============================================================================
// CompetitiveCoEvolution
///
/// Two populations with opposing objectives (predator / prey).
/// Predator fitness  = how well it exploits the prey.
/// Prey fitness      = how well it evades the predator.
///
/// @p predatorFitness(predator_genes, prey_genes) -> double
/// @p preyFitness(prey_genes, predator_genes)     -> double
// ============================================================================

class CompetitiveCoEvolution {
public:
    using FitnessFunc = std::function<double(const std::vector<double>&,
                                             const std::vector<double>&)>;

    explicit CompetitiveCoEvolution(const CoEvConfig& cfg) : cfg_(cfg) {
        unsigned seed = cfg_.seed == 0 ? std::random_device{}() : cfg_.seed;
        rng_.seed(seed);
    }

    /// @p predFit  fitness of predator given (predator, prey).
    /// @p preyFit  fitness of prey    given (prey,    predator).
    CoEvResult run(const FitnessFunc& predFit,
                   const FitnessFunc& preyFit);

private:
    struct Individual { std::vector<double> genes; double fit = -1e300; };
    using Pop = std::vector<Individual>;

    CoEvConfig   cfg_;
    std::mt19937 rng_;

    Pop initPop(int dim);
    Individual crossover(const Individual& a, const Individual& b);
    void mutate(Individual& ind);
    void clip(std::vector<double>& x) const;
    const Individual& tournament(const Pop& pop);
};

} // namespace coev
} // namespace ga
