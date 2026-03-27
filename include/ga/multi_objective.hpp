#pragma once
/// Multi-Objective Optimization
///
/// Implements:
///   - NSGA-II  (Non-dominated Sorting GA II – Deb et al., 2002)
///   - SPEA2    (Strength Pareto Evolutionary Algorithm 2)
///
/// Both algorithms maximise all objectives (negate if minimising).

#include <vector>
#include <functional>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ga {
namespace mo {

// ============================================================================
// MOConfig
// ============================================================================

struct MOConfig {
    int    populationSize = 100;
    int    generations    = 100;
    int    dimension      = 10;        ///< number of decision variables
    int    numObjectives  = 2;
    double crossoverRate  = 0.9;
    double mutationRate   = 0.1;
    double lowerBound     = -5.0;
    double upperBound     =  5.0;
    unsigned seed         = 0;
};

// ============================================================================
// MOIndividual – an individual with a vector of objective values
// ============================================================================

struct MOIndividual {
    std::vector<double> genes;
    std::vector<double> objectives;    ///< one per objective (higher = better)

    // NSGA-II attributes
    int    rank            = 0;
    double crowdingDistance = 0.0;

    // SPEA2 attributes
    double strength        = 0.0;
    double rawFitness      = 0.0;
    double density         = 0.0;
    double spea2Fitness    = 0.0;
};

// ============================================================================
// Pareto utilities
// ============================================================================

/// Returns true if a dominates b (all objectives of a >= b, at least one >).
inline bool dominates(const MOIndividual& a, const MOIndividual& b) {
    bool better = false;
    for (size_t i = 0; i < a.objectives.size(); ++i) {
        if (a.objectives[i] < b.objectives[i]) return false;
        if (a.objectives[i] > b.objectives[i]) better = true;
    }
    return better;
}

/// Extract the non-dominated (Pareto) front from a population.
std::vector<MOIndividual> paretoFront(const std::vector<MOIndividual>& pop);

// ============================================================================
// MOResult
// ============================================================================

struct MOResult {
    std::vector<MOIndividual> paretoFront;   ///< non-dominated solutions
    std::vector<double>       hypervolume;   ///< HV per generation (approximate)
};

// ============================================================================
// NSGA-II
// ============================================================================

class NSGAII {
public:
    using FitnessFunc = std::function<std::vector<double>(const std::vector<double>&)>;

    explicit NSGAII(const MOConfig& cfg) : cfg_(cfg) {
        unsigned seed = cfg_.seed == 0 ? std::random_device{}() : cfg_.seed;
        rng_.seed(seed);
    }

    MOResult run(const FitnessFunc& fitness);

private:
    MOConfig     cfg_;
    std::mt19937 rng_;

    std::vector<MOIndividual> initPop(const FitnessFunc& f);
    std::vector<MOIndividual> makeOffspring(const std::vector<MOIndividual>& pop,
                                            const FitnessFunc& f);

    // NSGA-II core
    void nonDominatedSort(std::vector<MOIndividual>& pop,
                          std::vector<std::vector<int>>& fronts);
    void crowdingDistanceAssignment(std::vector<MOIndividual>& pop,
                                    const std::vector<int>& front);
    const MOIndividual& tournamentSelect(const std::vector<MOIndividual>& pop);

    // Operators (SBX crossover + polynomial mutation for real-valued)
    std::pair<std::vector<double>,std::vector<double>>
        sbxCrossover(const std::vector<double>& p1,
                     const std::vector<double>& p2, double eta_c = 20.0);
    void polynomialMutation(std::vector<double>& x, double eta_m = 20.0);
    void clip(std::vector<double>& x) const;
};

// ============================================================================
// SPEA2
// ============================================================================

class SPEA2 {
public:
    using FitnessFunc = std::function<std::vector<double>(const std::vector<double>&)>;

    explicit SPEA2(const MOConfig& cfg) : cfg_(cfg) {
        unsigned seed = cfg_.seed == 0 ? std::random_device{}() : cfg_.seed;
        rng_.seed(seed);
    }

    MOResult run(const FitnessFunc& fitness);

private:
    MOConfig     cfg_;
    std::mt19937 rng_;

    std::vector<MOIndividual> initPop(const FitnessFunc& f);
    void assignFitness(std::vector<MOIndividual>& pop,
                       std::vector<MOIndividual>& archive);
    std::vector<MOIndividual> environmentalSelection(
        std::vector<MOIndividual>& pop,
        std::vector<MOIndividual>& archive);

    std::pair<std::vector<double>,std::vector<double>>
        sbxCrossover(const std::vector<double>& p1,
                     const std::vector<double>& p2, double eta_c = 20.0);
    void polynomialMutation(std::vector<double>& x, double eta_m = 20.0);
    void clip(std::vector<double>& x) const;
    const MOIndividual& binaryTournament(const std::vector<MOIndividual>& pool);
};

} // namespace mo
} // namespace ga
