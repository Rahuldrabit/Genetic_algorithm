#pragma once
/// Evolution Strategies: (μ+λ)-ES, (μ,λ)-ES, and CMA-ES.
///
/// All strategies work on real-valued vectors (std::vector<double>).
/// Fitness convention: higher is better (same as the rest of the framework).

#include <vector>
#include <functional>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ga {
namespace es {

// ============================================================================
// ESConfig
// ============================================================================

struct ESConfig {
    int    mu          = 10;    ///< number of parents selected each generation
    int    lambda      = 50;    ///< number of offspring produced per generation
    int    generations = 200;
    int    dimension   = 10;
    double lowerBound  = -5.0;
    double upperBound  =  5.0;
    double initSigma   = 0.5;   ///< initial step-size / mutation std-dev
    unsigned seed      = 0;
};

// ============================================================================
// ESResult
// ============================================================================

struct ESResult {
    std::vector<double> bestGenes;
    double bestFitness = -1e300;
    std::vector<double> bestHistory;
    std::vector<double> avgHistory;
};

// ============================================================================
// MuPlusLambdaES – (μ+λ): offspring + parents compete
// ============================================================================

class MuPlusLambdaES {
public:
    using FitnessFunc = std::function<double(const std::vector<double>&)>;

    explicit MuPlusLambdaES(const ESConfig& cfg) : cfg_(cfg) {
        unsigned seed = cfg_.seed == 0 ? std::random_device{}() : cfg_.seed;
        rng_.seed(seed);
    }

    ESResult run(const FitnessFunc& fitness);

private:
    struct Ind { std::vector<double> x; double sigma; double fit; };

    ESConfig     cfg_;
    std::mt19937 rng_;

    std::vector<Ind> initPop(const FitnessFunc& f);
    Ind mutate(const Ind& parent);
    void clip(std::vector<double>& x) const;
};

// ============================================================================
// MuCommaLambdaES – (μ,λ): only offspring survive (λ ≥ μ required)
// ============================================================================

class MuCommaLambdaES {
public:
    using FitnessFunc = std::function<double(const std::vector<double>&)>;

    explicit MuCommaLambdaES(const ESConfig& cfg) : cfg_(cfg) {
        if (cfg_.lambda < cfg_.mu)
            throw std::invalid_argument("(mu,lambda)-ES: lambda must be >= mu");
        unsigned seed = cfg_.seed == 0 ? std::random_device{}() : cfg_.seed;
        rng_.seed(seed);
    }

    ESResult run(const FitnessFunc& fitness);

private:
    struct Ind { std::vector<double> x; double sigma; double fit; };

    ESConfig     cfg_;
    std::mt19937 rng_;

    std::vector<Ind> initPop(const FitnessFunc& f);
    Ind mutate(const Ind& parent);
    void clip(std::vector<double>& x) const;
};

// ============================================================================
// CMAES – Covariance Matrix Adaptation Evolution Strategy
//
// Simplified single-population CMA-ES following the canonical
// Hansen & Ostermeier formulation.
// ============================================================================

class CMAES {
public:
    using FitnessFunc = std::function<double(const std::vector<double>&)>;

    explicit CMAES(const ESConfig& cfg);

    ESResult run(const FitnessFunc& fitness);

private:
    ESConfig     cfg_;
    std::mt19937 rng_;
    int          n_;        ///< problem dimension

    // CMA state
    std::vector<double> mean_;       ///< distribution mean
    double              sigma_;      ///< overall step size
    std::vector<double> pc_;         ///< evolution path for C
    std::vector<double> ps_;         ///< evolution path for sigma
    std::vector<std::vector<double>> C_;  ///< covariance matrix (n×n)
    std::vector<std::vector<double>> B_;  ///< eigenvectors
    std::vector<double>              D_;  ///< eigenvalues (sqrt)

    // Strategy parameters
    double mu_w_;
    double c_sigma_, d_sigma_, c_c_, c_1_, c_mu_;
    std::vector<double> weights_;
    int mu_int_;

    // helpers
    std::vector<double> sampleOffspring();
    void eigenDecompose();
    void clip(std::vector<double>& x) const;
    double norm(const std::vector<double>& v) const;
    std::vector<double> matVecMul(const std::vector<std::vector<double>>& M,
                                  const std::vector<double>& v) const;
    std::vector<double> outerAdd(const std::vector<double>& a,
                                 const std::vector<double>& b,
                                 double w) const;
};

} // namespace es
} // namespace ga
