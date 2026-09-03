/**
 * Comprehensive sanity tests for advanced GA features.
 * Tests CMA-ES, SPEA2, adaptive operators, hybrid optimization,
 * co-evolution, and evolution strategies.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <memory>

// Evolution Strategies
#include "ga/es/evolution_strategies.hpp"
#include "ga/es/cmaes.hpp"

// Multi-objective optimization
#include "ga/moea/spea2.hpp"

// Adaptive operators
#include "ga/adaptive/adaptive_policy.hpp"

// Hybrid optimization
#include "ga/hybrid/hybrid_optimizer.hpp"

// Co-evolution
#include "ga/coevolution/coevolution.hpp"

// Core components
#include "ga/core/individual.hpp"
#include "ga/config.hpp"

// Test result tracking
static int tests_passed = 0;
static int tests_failed = 0;
static constexpr double kPi = 3.14159265358979323846;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] " << msg << std::endl; \
            tests_failed++; \
            return false; \
        } \
    } while (0)

#define RUN_TEST(test_func) \
    do { \
        if (test_func()) { \
            std::cout << "[PASS] " << #test_func << std::endl; \
            tests_passed++; \
        } else { \
            tests_failed++; \
        } \
    } while (0)

// Simple fitness functions for testing
double sphere(const std::vector<double>& x) {
    double sum = 0.0;
    for (double v : x) {
        sum += v * v;
    }
    // Convert minimization to maximization
    return 1000.0 / (1.0 + sum);
}

double rastrigin(const std::vector<double>& x) {
    const double A = 10.0;
    double sum = A * static_cast<double>(x.size());
    for (double xi : x) {
        sum += xi * xi - A * std::cos(2.0 * kPi * xi);
    }
    // Convert minimization to maximization
    return 1000.0 / (1.0 + sum);
}

// Multi-objective test problems
std::vector<double> zdt1(const std::vector<double>& x) {
    if (x.empty()) return {0.0, 0.0};
    double f1 = x[0];
    double g = 0.0;
    for (size_t i = 1; i < x.size(); ++i) {
        g += x[i];
    }
    g = 1.0 + 9.0 * g / static_cast<double>(x.size() - 1);
    double h = 1.0 - std::sqrt(f1 / g);
    double f2 = g * h;
    return {f1, f2};
}

// =============================================================================
// TEST: Evolution Strategies (mu, lambda) and (mu + lambda)
// =============================================================================
bool test_evolution_strategies() {
    ga::es::EvolutionStrategyConfig cfg;
    cfg.mu = 5;
    cfg.lambda = 20;
    cfg.dimension = 5;
    cfg.generations = 30;
    cfg.lower = -5.0;
    cfg.upper = 5.0;
    cfg.sigma = 0.5;
    cfg.seed = 42;

    // Test comma strategy (mu, lambda)
    cfg.plusStrategy = false;
    ga::es::EvolutionStrategy es_comma(cfg);
    auto result_comma = es_comma.run(sphere);

    TEST_ASSERT(!result_comma.best.empty(), "ES comma strategy produced empty result");
    TEST_ASSERT(result_comma.best.size() == cfg.dimension, "ES comma strategy dimension mismatch");
    TEST_ASSERT(result_comma.bestFitness > 0.0, "ES comma strategy fitness should be positive");
    TEST_ASSERT(!result_comma.bestHistory.empty(), "ES comma strategy history should not be empty");

    // Test plus strategy (mu + lambda)
    cfg.plusStrategy = true;
    ga::es::EvolutionStrategy es_plus(cfg);
    auto result_plus = es_plus.run(sphere);

    TEST_ASSERT(!result_plus.best.empty(), "ES plus strategy produced empty result");
    TEST_ASSERT(result_plus.best.size() == cfg.dimension, "ES plus strategy dimension mismatch");
    TEST_ASSERT(result_plus.bestFitness > 0.0, "ES plus strategy fitness should be positive");
    TEST_ASSERT(!result_plus.bestHistory.empty(), "ES plus strategy history should not be empty");

    // Plus strategy should generally perform better or equal to comma
    TEST_ASSERT(result_plus.bestFitness >= result_comma.bestFitness * 0.5,
                "ES plus strategy should be competitive with comma strategy");

    return true;
}

// =============================================================================
// TEST: CMA-ES (Diagonal Covariance Matrix Adaptation)
// =============================================================================
bool test_cmaes() {
    ga::es::CmaEsConfig cfg;
    cfg.populationSize = 20;
    cfg.generations = 50;
    cfg.dimension = 5;
    cfg.lower = -5.0;
    cfg.upper = 5.0;
    cfg.sigma = 0.5;
    cfg.seed = 42;

    ga::es::DiagonalCmaEs cmaes(cfg);
    auto result = cmaes.run(sphere);

    TEST_ASSERT(!result.best.empty(), "CMA-ES produced empty result");
    TEST_ASSERT(result.best.size() == cfg.dimension, "CMA-ES dimension mismatch");
    TEST_ASSERT(result.bestFitness > 0.0, "CMA-ES fitness should be positive");
    TEST_ASSERT(!result.history.empty(), "CMA-ES history should not be empty");
    TEST_ASSERT(result.history.size() == cfg.generations, "CMA-ES history size mismatch");

    // Check that fitness improves over time
    double firstFitness = result.history.front();
    double lastFitness = result.history.back();
    TEST_ASSERT(lastFitness >= firstFitness * 0.9, "CMA-ES should show improvement");

    // Test with Rastrigin function (more challenging)
    auto result_rastrigin = cmaes.run(rastrigin);
    TEST_ASSERT(!result_rastrigin.best.empty(), "CMA-ES on Rastrigin produced empty result");
    TEST_ASSERT(result_rastrigin.bestFitness > 0.0, "CMA-ES on Rastrigin fitness should be positive");

    return true;
}

// =============================================================================
// TEST: SPEA2 (Strength Pareto Evolutionary Algorithm 2)
// =============================================================================
bool test_spea2() {
    ga::moea::Spea2 spea2;

    // Create a small test population with multi-objective evaluations
    std::vector<ga::Individual> population;
    for (int i = 0; i < 10; ++i) {
        ga::Individual ind;
        std::vector<double> x(5, 0.0);
        for (size_t j = 0; j < 5; ++j) {
            x[j] = static_cast<double>(i) / 10.0 + static_cast<double>(j) / 50.0;
        }
        ind.evaluation.objectives = zdt1(x);
        population.push_back(ind);
    }

    // Test strength fitness calculation
    auto fitness = spea2.strengthFitness(population);
    TEST_ASSERT(fitness.size() == population.size(), "SPEA2 fitness vector size mismatch");
    TEST_ASSERT(!fitness.empty(), "SPEA2 fitness should not be empty");

    // All fitness values should be non-negative
    for (double f : fitness) {
        TEST_ASSERT(f >= 0.0, "SPEA2 fitness should be non-negative");
    }

    // Test environmental selection
    std::size_t targetSize = 5;
    auto selected = spea2.environmentalSelect(population, targetSize);
    TEST_ASSERT(selected.size() == targetSize, "SPEA2 environmental selection size mismatch");

    // Test with empty population
    auto emptyFitness = spea2.strengthFitness({});
    TEST_ASSERT(emptyFitness.empty(), "SPEA2 fitness for empty population should be empty");

    auto emptySelected = spea2.environmentalSelect({}, 5);
    TEST_ASSERT(emptySelected.empty(), "SPEA2 selection from empty population should be empty");

    return true;
}

// =============================================================================
// TEST: Adaptive Rate Controller
// =============================================================================
bool test_adaptive_operators() {
    ga::adaptive::AdaptiveRateController controller(0.001, 0.6, 0.4, 0.95);

    ga::adaptive::AdaptiveRates rates;
    rates.mutationRate = 0.1;
    rates.crossoverRate = 0.8;

    // Test with low diversity and low improvement (should increase mutation)
    auto updated1 = controller.update(rates, 0.1, 0.0);
    TEST_ASSERT(updated1.mutationRate > rates.mutationRate,
                "Mutation rate should increase with low diversity");
    TEST_ASSERT(updated1.crossoverRate <= rates.crossoverRate,
                "Crossover rate should decrease with low diversity");

    // Test with high diversity and good improvement (should decrease mutation)
    auto updated2 = controller.update(rates, 0.5, 0.1);
    TEST_ASSERT(updated2.mutationRate < rates.mutationRate,
                "Mutation rate should decrease with high diversity");
    TEST_ASSERT(updated2.crossoverRate >= rates.crossoverRate,
                "Crossover rate should increase with high diversity");

    // Test boundary enforcement
    ga::adaptive::AdaptiveRates extremeRates;
    extremeRates.mutationRate = 0.001;
    extremeRates.crossoverRate = 0.95;

    auto bounded = controller.update(extremeRates, 0.5, 0.1);
    TEST_ASSERT(bounded.mutationRate >= 0.001, "Mutation rate should not go below minimum");
    TEST_ASSERT(bounded.crossoverRate <= 0.95, "Crossover rate should not exceed maximum");

    return true;
}

// =============================================================================
// TEST: Hybrid Optimization (GA + Local Search)
// =============================================================================
bool test_hybrid_optimization() {
    ga::Config cfg;
    cfg.populationSize = 20;
    cfg.generations = 20;
    cfg.dimension = 5;
    cfg.bounds = {-5.0, 5.0};
    cfg.mutationRate = 0.1;
    cfg.crossoverRate = 0.8;
    cfg.seed = 42;

    ga::hybrid::HybridOptimizer hybrid(cfg);

    // Simple hill-climbing local search
    auto localSearch = [](std::vector<double>& x) {
        double stepSize = 0.1;
        for (int iter = 0; iter < 10; ++iter) {
            double currentFitness = sphere(x);
            bool improved = false;
            for (size_t i = 0; i < x.size(); ++i) {
                // Try small positive perturbation
                x[i] += stepSize;
                double newFitness = sphere(x);
                if (newFitness > currentFitness) {
                    currentFitness = newFitness;
                    improved = true;
                } else {
                    // Try negative perturbation
                    x[i] -= 2.0 * stepSize;
                    newFitness = sphere(x);
                    if (newFitness > currentFitness) {
                        currentFitness = newFitness;
                        improved = true;
                    } else {
                        x[i] += stepSize; // Restore
                    }
                }
            }
            if (!improved) {
                stepSize *= 0.5; // Reduce step size
            }
        }
    };

    auto result = hybrid.run(sphere, localSearch, 3);

    TEST_ASSERT(!result.bestGenes.empty(), "Hybrid optimizer produced empty result");
    TEST_ASSERT(result.bestGenes.size() == static_cast<size_t>(cfg.dimension), "Hybrid optimizer dimension mismatch");
    TEST_ASSERT(result.bestFitness > 0.0, "Hybrid optimizer fitness should be positive");
    TEST_ASSERT(!result.bestHistory.empty(), "Hybrid optimizer history should not be empty");

    // Test without local search
    auto resultNoLS = hybrid.run(sphere, nullptr, 0);
    TEST_ASSERT(!resultNoLS.bestGenes.empty(), "Hybrid optimizer without LS produced empty result");

    return true;
}

// =============================================================================
// TEST: Co-evolution Engine
// =============================================================================
bool test_coevolution() {
    ga::coevolution::CoevolutionConfig cfg;
    cfg.generations = 10;
    cfg.seed = 42;

    ga::coevolution::CoevolutionEngine engine(cfg);

    // Create two populations
    ga::coevolution::CoevolutionEngine::Populations populations(2);

    // Initialize first population
    for (int i = 0; i < 5; ++i) {
        ga::Individual ind;
        ind.evaluation.objectives = {static_cast<double>(i), 0.0};
        populations[0].push_back(ind);
    }

    // Initialize second population
    for (int i = 0; i < 5; ++i) {
        ga::Individual ind;
        ind.evaluation.objectives = {static_cast<double>(i * 2), 0.0};
        populations[1].push_back(ind);
    }

    // Define cross-population evaluation
    auto evaluate = [](ga::coevolution::CoevolutionEngine::Populations& pops) {
        for (auto& pop : pops) {
            for (auto& ind : pop) {
                // Simple fitness based on first objective
                if (!ind.evaluation.objectives.empty()) {
                    ind.evaluation.objectives[1] = 100.0 / (1.0 + ind.evaluation.objectives[0]);
                }
            }
        }
    };

    // Define simple reproduction
    auto reproduce = [](ga::coevolution::CoevolutionEngine::Populations& pops, std::mt19937& rng) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (auto& pop : pops) {
            for (auto& ind : pop) {
                if (!ind.evaluation.objectives.empty()) {
                    // Simple mutation
                    ind.evaluation.objectives[0] += dist(rng) * 0.5 - 0.25;
                }
            }
        }
    };

    auto result = engine.run(populations, evaluate, reproduce);

    TEST_ASSERT(result.size() == 2, "Co-evolution should maintain both populations");
    TEST_ASSERT(!result[0].empty(), "First population should not be empty");
    TEST_ASSERT(!result[1].empty(), "Second population should not be empty");
    TEST_ASSERT(result[0].size() == populations[0].size(),
                "First population size should be maintained");
    TEST_ASSERT(result[1].size() == populations[1].size(),
                "Second population size should be maintained");

    return true;
}

// =============================================================================
// TEST: Integration - All features work together
// =============================================================================
bool test_integration() {
    // Test that we can use multiple advanced features in sequence

    // 1. Run CMA-ES
    ga::es::CmaEsConfig cmaConfig;
    cmaConfig.populationSize = 10;
    cmaConfig.generations = 10;
    cmaConfig.dimension = 3;
    cmaConfig.seed = 42;

    ga::es::DiagonalCmaEs cmaes(cmaConfig);
    auto cmaResult = cmaes.run(sphere);
    TEST_ASSERT(cmaResult.bestFitness > 0.0, "CMA-ES integration test failed");

    // 2. Use adaptive operators
    ga::adaptive::AdaptiveRateController controller;
    ga::adaptive::AdaptiveRates rates;
    auto updated = controller.update(rates, 0.3, 0.05);
    TEST_ASSERT(updated.mutationRate > 0.0, "Adaptive operators integration test failed");

    // 3. SPEA2 for multi-objective
    ga::moea::Spea2 spea2;
    std::vector<ga::Individual> testPop(3);
    for (size_t i = 0; i < 3; ++i) {
        testPop[i].evaluation.objectives = {static_cast<double>(i),
                                            static_cast<double>(3 - i)};
    }
    auto fitness = spea2.strengthFitness(testPop);
    TEST_ASSERT(fitness.size() == 3, "SPEA2 integration test failed");

    // 4. Evolution strategies
    ga::es::EvolutionStrategyConfig esConfig;
    esConfig.mu = 3;
    esConfig.lambda = 10;
    esConfig.dimension = 3;
    esConfig.generations = 10;
    esConfig.seed = 42;

    ga::es::EvolutionStrategy es(esConfig);
    auto esResult = es.run(sphere);
    TEST_ASSERT(esResult.bestFitness > 0.0, "ES integration test failed");

    return true;
}

// =============================================================================
// Main test runner
// =============================================================================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Advanced GA Features Sanity Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    RUN_TEST(test_evolution_strategies);
    RUN_TEST(test_cmaes);
    RUN_TEST(test_spea2);
    RUN_TEST(test_adaptive_operators);
    RUN_TEST(test_hybrid_optimization);
    RUN_TEST(test_coevolution);
    RUN_TEST(test_integration);

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;

    if (tests_failed == 0) {
        std::cout << std::endl;
        std::cout << "ALL PASS" << std::endl;
        return 0;
    } else {
        std::cout << std::endl;
        std::cout << "SOME TESTS FAILED" << std::endl;
        return 1;
    }
}
