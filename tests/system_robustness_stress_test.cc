/**
 * Comprehensive Robustness and Stress Test Suite
 * Validates the entire genetic algorithm & metaheuristics framework under:
 *  1. Challenging mathematical benchmark landscapes (Sphere, Rosenbrock, Rastrigin, Ackley, Griewank)
 *  2. Extreme boundary & edge cases (1D, high-D, minimal population, negative bounds, flat fitness)
 *  3. Dynamic environment tracking & adaptation under sudden landscape shifts
 *  4. Infeasibility recovery & constraint enforcement (Deb's rules, adaptive penalties, repair)
 *  5. Concurrent multi-threaded execution safety (parallel runs across independent threads)
 *  6. Fuzzy inference boundary and degenerate input resilience
 *  7. Shake perturbation and adaptive controller stability
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "ga/metaheuristics.hpp"

namespace {

int tests_passed = 0;
int tests_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            tests_failed++; \
            return false; \
        } \
    } while (0)

#define RUN_TEST(fn) \
    do { \
        std::cout << "[RUN ] " << #fn << "..." << std::flush; \
        if (fn()) { \
            std::cout << " [PASS]" << std::endl; \
            tests_passed++; \
        } else { \
            tests_failed++; \
        } \
    } while (0)

constexpr double kPi = 3.14159265358979323846;

// =============================================================================
// Benchmark Functions (All converted to MAXIMIZATION: higher is better)
// =============================================================================

// 1. Sphere: smooth, unimodal. Max at 0
double sphereFitness(const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) s += v * v;
    return 1000.0 / (1.0 + s);
}

// 2. Rosenbrock: non-separable, narrow parabolic valley. Max at (1,...,1)
double rosenbrockFitness(const std::vector<double>& x) {
    double s = 0.0;
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        double d1 = x[i + 1] - x[i] * x[i];
        double d2 = 1.0 - x[i];
        s += 100.0 * d1 * d1 + d2 * d2;
    }
    return 1000.0 / (1.0 + s);
}

// 3. Rastrigin: highly multimodal with cosine ripples. Max at 0
double rastriginFitness(const std::vector<double>& x) {
    double s = 10.0 * static_cast<double>(x.size());
    for (double v : x) {
        s += (v * v - 10.0 * std::cos(2.0 * kPi * v));
    }
    return 1000.0 / (1.0 + s);
}

// 4. Ackley: sharp central hole with nearly flat surrounding plateau. Max at 0
double ackleyFitness(const std::vector<double>& x) {
    double sumSq = 0.0;
    double sumCos = 0.0;
    const double n = static_cast<double>(x.size());
    for (double v : x) {
        sumSq += v * v;
        sumCos += std::cos(2.0 * kPi * v);
    }
    double term1 = -20.0 * std::exp(-0.2 * std::sqrt(sumSq / n));
    double term2 = -std::exp(sumCos / n);
    double cost = term1 + term2 + 20.0 + std::exp(1.0);
    return 1000.0 / (1.0 + cost);
}

// 5. Griewank: product-coupled widespread local traps. Max at 0
double griewankFitness(const std::vector<double>& x) {
    double sumSq = 0.0;
    double prodCos = 1.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        sumSq += (x[i] * x[i]) / 4000.0;
        prodCos *= std::cos(x[i] / std::sqrt(static_cast<double>(i + 1)));
    }
    double cost = sumSq - prodCos + 1.0;
    return 1000.0 / (1.0 + cost);
}

// =============================================================================
// Category 1: Challenging Benchmark Landscape Robustness
// =============================================================================

bool test_benchmark_landscapes_pso() {
    // Test PSO across Sphere, Rosenbrock, Rastrigin, Ackley
    auto rSphere = ga::api::runPso(sphereFitness, ga::pso::PsoVariant::GlobalBest, 4, 30, {-5.0, 5.0}, 42, 25);
    TEST_ASSERT(rSphere.bestFitness > 10.0, "PSO failed on Sphere");

    auto rRosen = ga::api::runPso(rosenbrockFitness, ga::pso::PsoVariant::Constriction, 3, 40, {-3.0, 3.0}, 42, 30);
    TEST_ASSERT(rRosen.bestFitness > 0.1, "PSO failed on Rosenbrock");

    auto rRastrigin = ga::api::runClpso(rastriginFitness, 4, 40, {-5.12, 5.12}, 42, 30);
    TEST_ASSERT(rRastrigin.bestFitness > 5.0, "CLPSO failed on Rastrigin");

    auto rAckley = ga::api::runPso(ackleyFitness, ga::pso::PsoVariant::FullyInformed, 3, 30, {-10.0, 10.0}, 42, 25);
    TEST_ASSERT(rAckley.bestFitness > 2.0, "PSO failed on Ackley");
    return true;
}

bool test_benchmark_landscapes_gso_and_gsa() {
    auto rGso = ga::api::runGso(sphereFitness, ga::gso::GsoVariant::AdaptiveStep, 3, 30, {-5.0, 5.0}, 42, 25);
    TEST_ASSERT(rGso.bestFitness > 10.0, "GSO adaptive failed on Sphere");

    auto rGsa = ga::api::runGsa(griewankFitness, 3, 30, {-10.0, 10.0}, 42, 25);
    TEST_ASSERT(rGsa.bestFitness > 1.0, "GSA failed on Griewank");
    return true;
}

bool test_benchmark_landscapes_island_and_heuristics() {
    // Island Model on Rastrigin
    auto rIsland = ga::api::runIslandModel(rastriginFitness, 4, 15, 3, 20, {-5.12, 5.12}, 42, ga::algorithms::IslandTopology::Ring);
    TEST_ASSERT(rIsland.bestFitness > 5.0, "Island Model failed on Rastrigin");

    // Simulated Annealing on Ackley
    auto rSa = ga::api::runSimulatedAnnealing(ackleyFitness, 3, 50, {-5.0, 5.0}, 100.0, 0.90, 42);
    TEST_ASSERT(rSa.bestFitness > 1.0, "Simulated Annealing failed on Ackley");

    // Hill Climbing on Sphere
    auto rHc = ga::api::runHillClimbing(sphereFitness, 3, 50, {-5.0, 5.0}, 0.05, 3, 42);
    TEST_ASSERT(rHc.bestFitness > 10.0, "Hill Climbing failed on Sphere");
    return true;
}

// =============================================================================
// Category 2: Boundary & Extreme Input Handling (Defensive Verification)
// =============================================================================

bool test_edge_case_1d_problem() {
    // 1-Dimensional optimization test
    auto r1d = ga::api::runPso(sphereFitness, ga::pso::PsoVariant::GlobalBest, 1, 20, {-10.0, 10.0}, 42, 10);
    TEST_ASSERT(r1d.bestSolution.size() == 1, "1D solution size mismatch");
    TEST_ASSERT(r1d.bestFitness > 50.0, "1D sphere should converge tightly");
    TEST_ASSERT(std::abs(r1d.bestSolution[0]) < 0.5, "1D optimum near zero");
    return true;
}

bool test_edge_case_high_dimensional() {
    // 50-Dimensional optimization scaling test
    const std::size_t dim = 50;
    auto rHighD = ga::api::runPso(sphereFitness, ga::pso::PsoVariant::Constriction, dim, 15, {-5.0, 5.0}, 42, 20);
    TEST_ASSERT(rHighD.bestSolution.size() == dim, "High-D dimension mismatch");
    TEST_ASSERT(rHighD.bestFitness > 0.0, "High-D fitness must be strictly positive");
    TEST_ASSERT(!rHighD.bestHistory.empty(), "High-D history must not be empty");
    return true;
}

bool test_edge_case_minimal_iterations_and_pop() {
    // Edge case: minimal population (5) and minimal generations (2)
    auto rMin = ga::api::runGeneticAlgorithm(sphereFitness, 2, 2, {-1.0, 1.0}, 42, 5);
    TEST_ASSERT(rMin.bestSolution.size() == 2, "Minimal run solution size mismatch");
    TEST_ASSERT(rMin.generations == 2, "Generations count mismatch");
    TEST_ASSERT(rMin.evaluations > 0, "Evaluations should be recorded");
    return true;
}

bool test_edge_case_asymmetric_negative_bounds() {
    // Optimum of shifted sphere at (-50, -50)
    auto shiftedSphere = [](const std::vector<double>& x) {
        double s = 0.0;
        for (double v : x) {
            double diff = v - (-50.0);
            s += diff * diff;
        }
        return 1000.0 / (1.0 + s);
    };

    ga::Bounds negBounds{-100.0, -20.0};
    auto rNeg = ga::api::runPso(shiftedSphere, ga::pso::PsoVariant::GlobalBest, 2, 30, negBounds, 42, 25);
    TEST_ASSERT(rNeg.bestSolution[0] >= negBounds.lower && rNeg.bestSolution[0] <= negBounds.upper, "Bounds violation");
    TEST_ASSERT(rNeg.bestSolution[1] >= negBounds.lower && rNeg.bestSolution[1] <= negBounds.upper, "Bounds violation");
    TEST_ASSERT(rNeg.bestFitness > 10.0, "Failed to optimize in negative bounds region");
    return true;
}

bool test_edge_case_flat_fitness_landscape() {
    // Completely flat landscape: no gradient, zero variance
    auto flatFitness = [](const std::vector<double>&) {
        return 42.0;
    };

    // Ensure algorithms handle zero variance without dividing by zero, crashing, or producing NaN
    auto rFlatPso = ga::api::runPso(flatFitness, ga::pso::PsoVariant::GlobalBest, 3, 10, {-5.0, 5.0}, 42, 10);
    TEST_ASSERT(!std::isnan(rFlatPso.bestFitness), "Flat fitness produced NaN in PSO");
    TEST_ASSERT(std::abs(rFlatPso.bestFitness - 42.0) < 1e-6, "Flat fitness value mismatch");

    auto rFlatGso = ga::api::runGso(flatFitness, ga::gso::GsoVariant::Standard, 3, 10, {-5.0, 5.0}, 42, 10);
    TEST_ASSERT(!std::isnan(rFlatGso.bestFitness), "Flat fitness produced NaN in GSO");

    auto rFlatIsland = ga::api::runIslandModel(flatFitness, 2, 10, 3, 10, {-5.0, 5.0}, 42);
    TEST_ASSERT(!std::isnan(rFlatIsland.bestFitness), "Flat fitness produced NaN in Island Model");
    return true;
}

bool test_edge_case_negative_fitness_range() {
    // Objective function returning strictly negative values
    auto negativeSphere = [](const std::vector<double>& x) {
        double s = 0.0;
        for (double v : x) s += v * v;
        return -100.0 - s; // Max is -100.0 at 0
    };

    auto rNegFit = ga::api::runPso(negativeSphere, ga::pso::PsoVariant::GlobalBest, 3, 20, {-5.0, 5.0}, 42, 20);
    TEST_ASSERT(rNegFit.bestFitness <= -100.0, "Best fitness should be <= -100.0");
    TEST_ASSERT(rNegFit.bestFitness > -150.0, "Failed to maximize negative fitness");
    return true;
}

// =============================================================================
// Category 3: Dynamic GA Under Abrupt Environmental Shift
// =============================================================================

bool test_dynamic_environment_shift_adaptation() {
    // Target shifts from (0, 0) to (3, 3) at evaluation call 400
    std::atomic<int> evalCount{0};
    auto dynamicLandscape = [&evalCount](const std::vector<double>& x) {
        int count = evalCount.fetch_add(1);
        double target = (count < 400) ? 0.0 : 3.0;
        double s = 0.0;
        for (double v : x) {
            double d = v - target;
            s += d * d;
        }
        return 1000.0 / (1.0 + s);
    };

    ga::adaptive::DynamicGAConfig cfg;
    cfg.search.dimension = 2;
    cfg.search.iterations = 35;
    cfg.search.populationSize = 25;
    cfg.search.bounds = {-5.0, 5.0};
    cfg.search.seed = 42;
    cfg.stagnationWindow = 3;
    cfg.strategy = ga::adaptive::DynamicStrategy::Hybrid;
    cfg.immigrantRatio = 0.25;
    cfg.hypermutationFactor = 5.0;
    cfg.detectEnvironmentChange = true;

    ga::adaptive::DynamicGAOptimizer dyn(cfg);
    auto res = dyn.optimize(dynamicLandscape);

    TEST_ASSERT(res.bestSolution.size() == 2, "Dynamic GA solution dimension mismatch");
    TEST_ASSERT(res.bestFitness > 0.0, "Dynamic GA fitness must be positive");
    // After shift to 3.0, the best solution should adapt towards 3.0
    double distToNewTarget = std::sqrt(std::pow(res.bestSolution[0] - 3.0, 2) + std::pow(res.bestSolution[1] - 3.0, 2));
    TEST_ASSERT(distToNewTarget < 2.0, "Dynamic GA failed to track shifted optimum");
    return true;
}

// =============================================================================
// Category 4: Constraint-Handling & Infeasibility Recovery
// =============================================================================

bool test_infeasibility_recovery_and_deb_rules() {
    using namespace ga::constraints;

    // Constraint: Point must lie inside a small hypersphere of radius 1.0 centered at (2.0, 2.0)
    // AND sum of elements >= 3.0
    ConstraintSet cset;
    cset.hard.push_back([](const std::vector<double>& g) {
        double d = std::sqrt(std::pow(g[0] - 2.0, 2) + std::pow(g[1] - 2.0, 2));
        return d <= 1.0;
    });
    cset.hard.push_back([](const std::vector<double>& g) {
        return (g[0] + g[1]) >= 3.0;
    });

    // Seed population with completely INFEASIBLE solutions (at origin 0,0 where constraints are violated)
    ga::metaheuristics::SeedPopulation infeasibleSeeds = {
        {0.0, 0.0},
        {-1.0, -1.0},
        {-2.0, -2.0},
        {5.0, 5.0}
    };

    ConstrainedOptimizerConfig cfg;
    cfg.search.dimension = 2;
    cfg.search.iterations = 30;
    cfg.search.populationSize = 35;
    cfg.search.bounds = {-5.0, 5.0};
    cfg.search.seed = 42;
    cfg.useDebFeasibility = true;
    cfg.useAdaptivePenalty = true;

    ConstrainedOptimizer opt(cfg, cset);
    auto res = opt.optimize(sphereFitness, infeasibleSeeds);

    // Verify final solution is strictly feasible
    TEST_ASSERT(isFeasible(res.bestSolution, cset), "Constrained optimizer failed to produce a feasible solution");
    double d = std::sqrt(std::pow(res.bestSolution[0] - 2.0, 2) + std::pow(res.bestSolution[1] - 2.0, 2));
    TEST_ASSERT(d <= 1.0001, "Distance constraint violated");
    TEST_ASSERT((res.bestSolution[0] + res.bestSolution[1]) >= 2.9999, "Sum constraint violated");
    return true;
}

bool test_repair_operators_stress() {
    using namespace ga::constraints;

    // Simplex repair stress: extreme positive/negative values
    std::vector<double> simplexGenes = {-5.0, 0.0, 10.0, 90.0, -100.0};
    repairSimplex(simplexGenes, 1.0);
    double sum = std::accumulate(simplexGenes.begin(), simplexGenes.end(), 0.0);
    TEST_ASSERT(std::abs(sum - 1.0) < 1e-9, "Simplex repair failed sum condition");
    for (double g : simplexGenes) {
        TEST_ASSERT(g >= 0.0, "Simplex components must be non-negative");
    }

    // Linear inequality repair: a^T x <= b
    // Constraint: 2*x0 + 3*x1 <= 5. Test point: (5, 5) -> 25 > 5
    std::vector<double> linearGenes = {5.0, 5.0};
    std::vector<double> a = {2.0, 3.0};
    double b = 5.0;
    repairLinearInequality(linearGenes, a, b);
    double dot = 2.0 * linearGenes[0] + 3.0 * linearGenes[1];
    TEST_ASSERT(dot <= 5.0001, "Linear inequality projection failed");

    // Permutation repair stress: completely corrupted permutation [7, 7, 7, 7, 7]
    std::vector<int> corruptedPerm = {7, 7, 7, 7, 7};
    repairPermutation(corruptedPerm);
    TEST_ASSERT(corruptedPerm.size() == 5, "Permutation size changed");
    std::vector<bool> seen(5, false);
    for (int p : corruptedPerm) {
        TEST_ASSERT(p >= 0 && p < 5, "Invalid element in repaired permutation");
        seen[p] = true;
    }
    for (bool s : seen) {
        TEST_ASSERT(s, "Missing elements in repaired permutation");
    }
    return true;
}

// =============================================================================
// Category 5: Multi-Threaded Concurrency Safety
// =============================================================================

bool test_concurrent_multithreaded_execution() {
    // Launch 8 concurrent threads simultaneously optimizing different functions
    constexpr int kNumThreads = 8;
    std::vector<std::future<bool>> futures;

    for (int t = 0; t < kNumThreads; ++t) {
        futures.push_back(std::async(std::launch::async, [t]() {
            unsigned seed = static_cast<unsigned>(100 + t * 37);
            
            // Thread t calls a distinct algorithm
            if (t % 4 == 0) {
                auto r = ga::api::runPso(sphereFitness, ga::pso::PsoVariant::GlobalBest, 3, 20, {-5.0, 5.0}, seed, 15);
                return r.bestFitness > 0.0 && r.bestSolution.size() == 3;
            } else if (t % 4 == 1) {
                auto r = ga::api::runSimulatedAnnealing(ackleyFitness, 3, 30, {-5.0, 5.0}, 50.0, 0.92, seed);
                return r.bestFitness > 0.0 && r.bestSolution.size() == 3;
            } else if (t % 4 == 2) {
                auto r = ga::api::runHillClimbing(sphereFitness, 3, 30, {-5.0, 5.0}, 0.05, 1, seed);
                return r.bestFitness > 0.0 && r.bestSolution.size() == 3;
            } else {
                auto r = ga::api::runIslandModel(sphereFitness, 2, 10, 3, 10, {-5.0, 5.0}, seed);
                return r.bestFitness > 0.0 && r.bestSolution.size() == 3;
            }
        }));
    }

    for (std::size_t i = 0; i < futures.size(); ++i) {
        bool ok = futures[i].get();
        TEST_ASSERT(ok, "Concurrent thread execution failed or crashed");
    }
    return true;
}

// =============================================================================
// Category 6: Fuzzy Subsystem Boundary & Resiliency Tests
// =============================================================================

bool test_fuzzy_boundary_and_degeneracy() {
    // 1. Extreme inputs outside defined domain
    ga::fuzzy::LinguisticVariable lv("error", ga::fuzzy::Interval{-10.0, 10.0});
    lv.addTerm("zero", ga::fuzzy::makeTriangularMF(-1.0, 0.0, 1.0));
    lv.addTerm("positive", ga::fuzzy::makeTrapezoidalMF(0.0, 2.0, 10.0, 15.0));

    // Fuzzify far beyond defined range
    auto fFarNeg = lv.fuzzify(-100.0);
    TEST_ASSERT(fFarNeg["zero"] == 0.0, "Far negative should yield 0 membership");

    auto fFarPos = lv.fuzzify(100.0);
    TEST_ASSERT(fFarPos["positive"] == 0.0, "Far positive should yield 0 membership");

    // 2. Mamdani evaluation with untriggered rules
    ga::fuzzy::MamdaniSystem mamdani;
    mamdani.addInput("speed", 0.0, 100.0);
    mamdani.addInputTerm("speed", "slow", ga::fuzzy::makeTriangularMF(-10.0, 0.0, 50.0));
    mamdani.addInputTerm("speed", "fast", ga::fuzzy::makeTriangularMF(50.0, 100.0, 150.0));

    mamdani.addOutput("brake", 0.0, 1.0);
    mamdani.addOutputTerm("brake", "light", ga::fuzzy::makeTriangularMF(0.0, 0.2, 0.5));
    mamdani.addOutputTerm("brake", "hard", ga::fuzzy::makeTriangularMF(0.5, 0.8, 1.0));

    mamdani.addRule("IF speed IS fast THEN brake IS hard");

    // Evaluate input where rule is activated
    double brakeVal = mamdani.evaluateSingle("brake", {{"speed", 80.0}});
    TEST_ASSERT(brakeVal >= 0.0 && brakeVal <= 1.0, "Mamdani output must be within output bounds");

    // Evaluate input where no rules trigger
    double defaultBrake = mamdani.evaluateSingle("brake", {{"speed", 0.0}});
    TEST_ASSERT(!std::isnan(defaultBrake), "Untriggered Mamdani evaluation must not yield NaN");
    TEST_ASSERT(defaultBrake >= 0.0 && defaultBrake <= 1.0, "Fallback output within bounds");

    // 3. Sugeno evaluation boundary
    ga::fuzzy::SugenoSystem sugeno;
    sugeno.addInput("temperature", 0.0, 100.0);
    sugeno.addInputTerm("temperature", "cold", ga::fuzzy::makeTriangularMF(-10.0, 0.0, 50.0));
    sugeno.addOutput("power", 0.75);
    sugeno.addRule(ga::fuzzy::SugenoRule({{"temperature", "cold", false}}, {{"power", 0.75, {}}}));

    double powerVal = sugeno.evaluateSingle("power", {{"temperature", 20.0}});
    TEST_ASSERT(std::abs(powerVal - 0.75) < 1e-6, "Sugeno constant output mismatch");
    return true;
}

// =============================================================================
// Category 7: Shake Framework and Hybrid Pipeline Stability
// =============================================================================

bool test_shake_and_hybrid_stability() {
    // Test Shake operators: uniform, gaussian, cauchy, levy, opposition
    std::vector<double> candidate = {1.0, 2.0, 3.0};
    ga::Bounds bounds{-5.0, 5.0};
    std::mt19937 rng(42);

    for (int i = 0; i < 20; ++i) {
        auto perturbed = candidate;
        ga::shake::gaussianShake(perturbed, bounds, 0.1, rng);
        TEST_ASSERT(perturbed.size() == candidate.size(), "Perturbed dimension mismatch");
        for (double v : perturbed) {
            TEST_ASSERT(v >= bounds.lower && v <= bounds.upper, "Perturbed element out of bounds");
        }

        ga::shake::cauchyShake(perturbed, bounds, 0.05, rng);
        for (double v : perturbed) {
            TEST_ASSERT(v >= bounds.lower && v <= bounds.upper, "Cauchy perturbed out of bounds");
        }

        ga::shake::levyFlightShake(perturbed, bounds, 0.05, 1.5, rng);
        for (double v : perturbed) {
            TEST_ASSERT(v >= bounds.lower && v <= bounds.upper, "Levy perturbed out of bounds");
        }
    }

    // Test Interleaved Hybrid Optimizer stability
    ga::metaheuristics::SearchConfig sc;
    sc.dimension = 3;
    sc.iterations = 10;
    sc.populationSize = 15;
    sc.bounds = {-5.0, 5.0};
    sc.seed = 42;

    ga::hybrid::InterleavedHybridConfig ihCfg;
    ihCfg.search = sc;
    ihCfg.epochs = 2;

    ga::pso::PsoConfig psoCfg;
    psoCfg.search = sc;
    auto opt1 = std::make_shared<ga::pso::ParticleSwarmOptimizer>(psoCfg);

    ga::gso::GsoConfig gsoCfg;
    gsoCfg.search = sc;
    auto opt2 = std::make_shared<ga::gso::GlowwormSwarmOptimizer>(gsoCfg);

    ga::hybrid::InterleavedHybridOptimizer interleaved(opt1, opt2, ihCfg);
    auto res = interleaved.optimize(sphereFitness);
    TEST_ASSERT(res.bestSolution.size() == 3, "Interleaved hybrid solution size mismatch");
    TEST_ASSERT(res.bestFitness > 0.0, "Interleaved hybrid fitness must be positive");
    return true;
}

// =============================================================================
// Category 8: Advanced Architecture Robustness (Scale, Latency, Topologies)
// =============================================================================

bool test_real_time_sub_millisecond_budget() {
    // Requirements state: Real-time constraints (<1ms per generation)
    // Verify that Simulated Annealing and Hill Climbing execute with < 1ms per iteration
    const std::size_t iterations = 1000;
    
    // 1. Simulated Annealing
    auto t0 = std::chrono::high_resolution_clock::now();
    auto resSa = ga::api::runSimulatedAnnealing(sphereFitness, 5, iterations, {-5.0, 5.0}, 100.0, 0.95, 42);
    auto t1 = std::chrono::high_resolution_clock::now();
    double saTotalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double saMsPerIter = saTotalMs / static_cast<double>(iterations);
    
    TEST_ASSERT(resSa.bestFitness > 0.0, "SA failed to produce positive fitness");
    TEST_ASSERT(saMsPerIter < 1.0, "SA exceeded 1ms per iteration real-time budget");

    // 2. Hill Climbing
    auto t2 = std::chrono::high_resolution_clock::now();
    auto resHc = ga::api::runHillClimbing(sphereFitness, 5, iterations, {-5.0, 5.0}, 0.05, 1, 42);
    auto t3 = std::chrono::high_resolution_clock::now();
    double hcTotalMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double hcMsPerIter = hcTotalMs / static_cast<double>(iterations);

    TEST_ASSERT(resHc.bestFitness > 0.0, "HC failed to produce positive fitness");
    TEST_ASSERT(hcMsPerIter < 1.0, "HC exceeded 1ms per iteration real-time budget");

    return true;
}

bool test_extreme_scale_population_island_model() {
    // Requirements state: Very large populations (>10,000)
    // Run 5 islands with 2,000 individuals each = 10,000 individuals total
    ga::algorithms::IslandConfig cfg;
    cfg.numIslands = 5;
    cfg.islandPopulationSize = 2000; // 10,000 total individuals!
    cfg.iterations = 5;
    cfg.migrationInterval = 2;
    cfg.migrantsPerExchange = 10;
    cfg.topology = ga::algorithms::IslandTopology::Ring;
    cfg.policy = ga::algorithms::MigrationPolicy::BestToWorst;
    cfg.search.dimension = 4;
    cfg.search.bounds = {-5.0, 5.0};
    cfg.search.seed = 42;

    ga::algorithms::IslandModelOptimizer islandOpt(cfg);
    auto res = islandOpt.optimize(sphereFitness);

    TEST_ASSERT(res.bestSolution.size() == 4, "10,000-pop Island Model solution size mismatch");
    TEST_ASSERT(res.bestFitness > 10.0, "10,000-pop Island Model failed to converge on Sphere");
    TEST_ASSERT(res.evaluations >= 10000, "10,000-pop Island Model evaluation count mismatch");
    return true;
}

bool test_all_island_topologies_and_policies() {
    using namespace ga::algorithms;
    std::vector<IslandTopology> topologies = {
        IslandTopology::Ring,
        IslandTopology::Star,
        IslandTopology::FullyConnected,
        IslandTopology::RandomMesh
    };

    std::vector<MigrationPolicy> policies = {
        MigrationPolicy::BestToWorst,
        MigrationPolicy::RandomToWorst,
        MigrationPolicy::BestToRandom
    };

    for (auto topo : topologies) {
        for (auto pol : policies) {
            IslandConfig cfg;
            cfg.numIslands = 4;
            cfg.islandPopulationSize = 20;
            cfg.iterations = 6;
            cfg.migrationInterval = 2;
            cfg.migrantsPerExchange = 2;
            cfg.topology = topo;
            cfg.policy = pol;
            cfg.search.dimension = 3;
            cfg.search.bounds = {-5.0, 5.0};
            cfg.search.seed = 123;

            IslandModelOptimizer opt(cfg);
            auto res = opt.optimize(sphereFitness);
            TEST_ASSERT(res.bestSolution.size() == 3, "Topology test dimension mismatch");
            TEST_ASSERT(res.bestFitness > 0.0, "Topology test non-positive fitness");
        }
    }
    return true;
}

bool test_degenerate_and_boundary_guards() {
    // 1. Inverted bounds [5.0, -5.0] must throw std::invalid_argument defensively
    bool caughtInverted = false;
    try {
        ga::api::runPso(sphereFitness, ga::pso::PsoVariant::GlobalBest, 2, 5, {5.0, -5.0}, 42, 5);
    } catch (const std::invalid_argument&) {
        caughtInverted = true;
    }
    TEST_ASSERT(caughtInverted, "Inverted bounds failed to throw std::invalid_argument");

    // 2. Zero-width bounds [0.0, 0.0] must throw std::invalid_argument defensively
    bool caughtZeroWidth = false;
    try {
        ga::api::runPso(sphereFitness, ga::pso::PsoVariant::GlobalBest, 2, 5, {0.0, 0.0}, 42, 5);
    } catch (const std::invalid_argument&) {
        caughtZeroWidth = true;
    }
    TEST_ASSERT(caughtZeroWidth, "Zero-width bounds failed to throw std::invalid_argument");

    // 3. Zero dimension must throw std::invalid_argument defensively
    bool caughtZeroDim = false;
    try {
        ga::api::runPso(sphereFitness, ga::pso::PsoVariant::GlobalBest, 0, 5, {-1.0, 1.0}, 42, 5);
    } catch (const std::invalid_argument&) {
        caughtZeroDim = true;
    }
    TEST_ASSERT(caughtZeroDim, "Zero dimension failed to throw std::invalid_argument");

    // 4. Narrow epsilon bounds [-1e-5, 1e-5] must optimize without underflow or NaN
    ga::Bounds narrowBounds{-1e-5, 1e-5};
    auto rNarrow = ga::api::runPso(sphereFitness, ga::pso::PsoVariant::GlobalBest, 2, 5, narrowBounds, 42, 10);
    TEST_ASSERT(rNarrow.bestSolution.size() == 2, "Narrow bounds solution size mismatch");
    TEST_ASSERT(!std::isnan(rNarrow.bestFitness), "Narrow bounds resulted in NaN");
    TEST_ASSERT(rNarrow.bestSolution[0] >= narrowBounds.lower && rNarrow.bestSolution[0] <= narrowBounds.upper, "Narrow bounds violated");

    // 5. High-dimension scaling test (100 dimensions)
    auto rDimSingle = ga::api::runHillClimbing(sphereFitness, 100, 5, {-1.0, 1.0}, 0.01, 1, 42);
    TEST_ASSERT(rDimSingle.bestSolution.size() == 100, "100-dimension HC dimension mismatch");
    TEST_ASSERT(!std::isnan(rDimSingle.bestFitness), "100-dimension HC produced NaN");

    return true;
}

} // namespace

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "      SYSTEM ROBUSTNESS & STRESS TEST SUITE               " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Challenging Landscapes
    RUN_TEST(test_benchmark_landscapes_pso);
    RUN_TEST(test_benchmark_landscapes_gso_and_gsa);
    RUN_TEST(test_benchmark_landscapes_island_and_heuristics);

    // 2. Defensive Boundaries & Extremes
    RUN_TEST(test_edge_case_1d_problem);
    RUN_TEST(test_edge_case_high_dimensional);
    RUN_TEST(test_edge_case_minimal_iterations_and_pop);
    RUN_TEST(test_edge_case_asymmetric_negative_bounds);
    RUN_TEST(test_edge_case_flat_fitness_landscape);
    RUN_TEST(test_edge_case_negative_fitness_range);

    // 3. Dynamic Environment Shift
    RUN_TEST(test_dynamic_environment_shift_adaptation);

    // 4. Constraint-Handling & Infeasibility Recovery
    RUN_TEST(test_infeasibility_recovery_and_deb_rules);
    RUN_TEST(test_repair_operators_stress);

    // 5. Multi-Threaded Concurrency
    RUN_TEST(test_concurrent_multithreaded_execution);

    // 6. Fuzzy Resiliency
    RUN_TEST(test_fuzzy_boundary_and_degeneracy);

    // 7. Shake & Hybrids
    RUN_TEST(test_shake_and_hybrid_stability);

    // 8. Scale, Real-Time Latency, Topologies & Degenerate Bounds
    RUN_TEST(test_real_time_sub_millisecond_budget);
    RUN_TEST(test_extreme_scale_population_island_model);
    RUN_TEST(test_all_island_topologies_and_policies);
    RUN_TEST(test_degenerate_and_boundary_guards);

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "  Passed: " << tests_passed << ", Failed: " << tests_failed << std::endl;
    std::cout << "==========================================================" << std::endl;

    return tests_failed == 0 ? 0 : 1;
}
