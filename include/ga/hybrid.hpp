#pragma once
/// Hybrid Optimization
///
/// Combines GA with local search to accelerate convergence.
///
/// HillClimbing        – simple coordinate-wise hill-climbing (local search).
/// GAPSOStep           – one step of PSO update (for GA+PSO hybrids).
/// HybridConfig        – shared configuration.
/// applyLocalSearch    – apply N steps of hill-climbing to the best K genes.

#include <vector>
#include <functional>
#include <random>
#include <algorithm>
#include <cmath>

namespace ga {
namespace hybrid {

// ============================================================================
// HybridConfig
// ============================================================================

struct HybridConfig {
    int    localSearchIter  = 20;    ///< hill-climbing steps per call
    int    localSearchElite = 3;     ///< number of top individuals to polish
    double stepSize         = 0.05;  ///< initial hill-climbing step
    double stepDecay        = 0.9;   ///< step size decay per iteration
    double lowerBound       = -5.0;
    double upperBound       =  5.0;
    unsigned seed           = 0;
};

// ============================================================================
// HillClimbing – coordinate descent local search
// ============================================================================

class HillClimbing {
public:
    using FitnessFunc = std::function<double(const std::vector<double>&)>;

    explicit HillClimbing(const HybridConfig& cfg) : cfg_(cfg) {
        unsigned seed = cfg_.seed == 0 ? std::random_device{}() : cfg_.seed;
        rng_.seed(seed);
    }

    /// Improve @p genes in-place and return improved fitness.
    double improve(std::vector<double>& genes, const FitnessFunc& f);

private:
    HybridConfig cfg_;
    std::mt19937 rng_;
    void clip(std::vector<double>& x) const;
};

// ============================================================================
// PSOParticle – a single PSO particle
// ============================================================================

struct PSOParticle {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> bestPos;   ///< personal best
    double bestFit = -1e300;
    double fit     = -1e300;
};

// ============================================================================
// PSOConfig
// ============================================================================

struct PSOConfig {
    int    swarmSize   = 30;
    int    generations = 100;
    int    dimension   = 10;
    double omega       = 0.7;   ///< inertia weight
    double phiP        = 1.5;   ///< cognitive coefficient
    double phiG        = 1.5;   ///< social coefficient
    double vMax        = 1.0;   ///< max velocity
    double lowerBound  = -5.0;
    double upperBound  =  5.0;
    unsigned seed      = 0;
};

// ============================================================================
// PSOResult
// ============================================================================

struct PSOResult {
    std::vector<double> bestGenes;
    double bestFitness = -1e300;
    std::vector<double> bestHistory;
};

// ============================================================================
// PSO – Particle Swarm Optimiser (standalone + hybrid component)
// ============================================================================

class PSO {
public:
    using FitnessFunc = std::function<double(const std::vector<double>&)>;

    explicit PSO(const PSOConfig& cfg) : cfg_(cfg) {
        unsigned seed = cfg_.seed == 0 ? std::random_device{}() : cfg_.seed;
        rng_.seed(seed);
    }

    PSOResult run(const FitnessFunc& fitness);

private:
    PSOConfig    cfg_;
    std::mt19937 rng_;

    std::vector<PSOParticle> initSwarm(const FitnessFunc& f);
    void clip(std::vector<double>& x) const;
};

// ============================================================================
// applyLocalSearch – convenience wrapper used inside a GA loop
///
/// Applies hill-climbing to the top @p cfg.localSearchElite individuals
/// from @p pop (sorted by fitness descending).  @p pop elements must have
/// `.genes` (vector<double>) and `.fitness` (double) members.
// ============================================================================

template <typename Individual>
void applyLocalSearch(std::vector<Individual>& pop,
                      const std::function<double(const std::vector<double>&)>& f,
                      const HybridConfig& cfg)
{
    HillClimbing hc(cfg);

    // Sort descending by fitness
    std::sort(pop.begin(), pop.end(),
              [](const Individual& a, const Individual& b){
                  return a.fitness > b.fitness;
              });

    int n = std::min(cfg.localSearchElite, (int)pop.size());
    for (int i = 0; i < n; ++i) {
        pop[i].fitness = hc.improve(pop[i].genes, f);
    }
}

} // namespace hybrid
} // namespace ga
