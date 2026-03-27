#pragma once
/// Adaptive Genetic Operators
///
/// Automatically tunes crossover and mutation rates based on population
/// diversity (entropy / standard deviation) and convergence speed.
///
/// DiversityMonitor  – measures population diversity each generation.
/// AdaptiveRates     – adjusts crossoverRate and mutationRate accordingly.

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace ga {
namespace adaptive {

// ============================================================================
// DiversityMonitor
// ============================================================================

class DiversityMonitor {
public:
    /// Compute average standard deviation of genes across the population.
    /// @p pop  Each element is one individual's gene vector.
    static double geneStdDev(const std::vector<std::vector<double>>& pop) {
        if (pop.empty() || pop[0].empty()) return 0.0;
        const size_t n = pop.size();
        const size_t d = pop[0].size();
        double total = 0.0;
        for (size_t j = 0; j < d; ++j) {
            double mean = 0.0;
            for (size_t i = 0; i < n; ++i) mean += pop[i][j];
            mean /= n;
            double var = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double diff = pop[i][j] - mean;
                var += diff * diff;
            }
            total += std::sqrt(var / n);
        }
        return total / d;
    }

    /// Compute fitness variance across the population.
    static double fitnessVariance(const std::vector<double>& fitnesses) {
        if (fitnesses.empty()) return 0.0;
        double mean = 0.0;
        for (double f : fitnesses) mean += f;
        mean /= fitnesses.size();
        double var = 0.0;
        for (double f : fitnesses) { double d = f - mean; var += d * d; }
        return var / fitnesses.size();
    }
};

// ============================================================================
// AdaptiveRates
///
/// Strategy:
///   - When diversity is HIGH (population spread out) → reduce mutation,
///     increase crossover to exploit good solutions.
///   - When diversity is LOW (population converged) → increase mutation to
///     escape local optima, reduce crossover.
// ============================================================================

class AdaptiveRates {
public:
    struct Bounds {
        double minCrossover = 0.5;
        double maxCrossover = 0.95;
        double minMutation  = 0.01;
        double maxMutation  = 0.4;
    };

    explicit AdaptiveRates(double initCrossover = 0.8,
                           double initMutation  = 0.1,
                           Bounds bounds        = {})
        : crossoverRate_(initCrossover),
          mutationRate_(initMutation),
          bounds_(bounds) {}

    /// Update rates based on current population gene diversity.
    /// @p diversity  Normalised diversity in [0, 1] (0 = converged, 1 = random).
    void update(double diversity) {
        // When diversity → 0: increase mutation, decrease crossover.
        // When diversity → 1: decrease mutation, increase crossover.
        crossoverRate_ = bounds_.minCrossover
            + diversity * (bounds_.maxCrossover - bounds_.minCrossover);
        mutationRate_ = bounds_.maxMutation
            - diversity * (bounds_.maxMutation - bounds_.minMutation);
    }

    /// Update using raw gene matrix and reference max-diversity level.
    /// @p maxStdDev  Expected std-dev when population is fully random
    ///               (e.g. (upperBound - lowerBound) / 4).
    void updateFromPop(const std::vector<std::vector<double>>& pop,
                       double maxStdDev = 2.0) {
        double sd = DiversityMonitor::geneStdDev(pop);
        double div = std::min(1.0, sd / (maxStdDev + 1e-10));
        update(div);
    }

    double crossoverRate() const { return crossoverRate_; }
    double mutationRate()  const { return mutationRate_;  }

private:
    double crossoverRate_;
    double mutationRate_;
    Bounds bounds_;
};

// ============================================================================
// SelfAdaptiveMutationSigma
///
/// Each individual carries its own step-size σ (as in ES).
/// After mutation the σ is updated via the 1/5 success rule.
// ============================================================================

class SelfAdaptiveMutationSigma {
public:
    explicit SelfAdaptiveMutationSigma(double initSigma = 0.1,
                                       double tau       = 0.0,  ///< 0 = auto
                                       int    dimension = 10)
        : sigma_(initSigma), tau_(tau == 0.0 ? 1.0 / std::sqrt(2.0 * dimension) : tau) {}

    /// Mutate σ (log-normal self-adaptation).
    void adaptSigma(std::mt19937& rng) {
        std::normal_distribution<double> nd(0.0, 1.0);
        sigma_ *= std::exp(tau_ * nd(rng));
        sigma_ = std::max(1e-8, sigma_);
    }

    double sigma() const { return sigma_; }
    void setSigma(double s) { sigma_ = s; }

private:
    double sigma_;
    double tau_;
};

} // namespace adaptive
} // namespace ga
