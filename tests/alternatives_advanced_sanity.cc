/**
 * Sanity tests for alternatives and advanced strategies:
 * 1. Island Model (Distributed GA for large populations)
 * 2. Simpler Real-Time Heuristics (Simulated Annealing, Hill Climbing <1ms)
 * 3. Dynamic GA (Adaptive GA for dynamic/non-stationary problems)
 * 4. Constraint-Heavy Handling (Deb's rules, adaptive penalty, repair operators)
 */

#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "ga/metaheuristics.hpp"

namespace {

int tests_passed = 0;
int tests_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] " << msg << " at line " << __LINE__ << std::endl; \
            tests_failed++; \
            return false; \
        } \
    } while (0)

#define RUN_TEST(fn) \
    do { \
        if (fn()) { \
            std::cout << "[PASS] " << #fn << std::endl; \
            tests_passed++; \
        } else { \
            tests_failed++; \
        } \
    } while (0)

double sphere(const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) s += v * v;
    return 1000.0 / (1.0 + s);
}

// ---------------- 1. Island Model (Distributed GA) ----------------
bool test_island_model() {
    ga::algorithms::IslandConfig cfg;
    cfg.numIslands = 4;
    cfg.islandPopulationSize = 25; // 100 total individuals
    cfg.iterations = 20;
    cfg.migrationInterval = 5;
    cfg.migrantsPerExchange = 2;
    cfg.topology = ga::algorithms::IslandTopology::Ring;
    cfg.search.dimension = 4;
    cfg.search.bounds = {-5.0, 5.0};
    cfg.search.seed = 1234;

    ga::algorithms::IslandModelOptimizer opt(cfg);
    auto res = opt.optimize(sphere);

    TEST_ASSERT(res.bestSolution.size() == 4, "Solution dimension mismatch");
    TEST_ASSERT(res.bestFitness > 0.0, "Fitness must be positive");
    TEST_ASSERT(res.evaluations > 0, "Evaluations must be counted");
    TEST_ASSERT(res.bestHistory.size() == 20, "History size must match iterations");

    // Test runner
    auto runnerRes = ga::api::runIslandModel(sphere, 2, 20, 3, 10, {-5.0, 5.0}, 42);
    TEST_ASSERT(runnerRes.bestSolution.size() == 3, "Runner dimension mismatch");
    TEST_ASSERT(runnerRes.bestFitness > 0.0, "Runner fitness must be positive");
    return true;
}

// ---------------- 2. Simpler Real-Time Heuristics ----------------
bool test_simulated_annealing() {
    ga::metaheuristics::SimulatedAnnealingConfig cfg;
    cfg.search.dimension = 3;
    cfg.search.iterations = 50;
    cfg.search.bounds = {-5.0, 5.0};
    cfg.search.seed = 42;
    cfg.initialTemperature = 50.0;
    cfg.coolingRate = 0.90;

    ga::metaheuristics::SimulatedAnnealingOptimizer sa(cfg);
    auto res = sa.optimize(sphere);

    TEST_ASSERT(res.bestSolution.size() == 3, "SA dimension mismatch");
    TEST_ASSERT(res.bestFitness > 0.0, "SA fitness must be positive");
    TEST_ASSERT(!res.bestHistory.empty(), "SA history must not be empty");

    // Test runner
    auto runnerRes = ga::api::runSimulatedAnnealing(sphere, 3, 30, {-5.0, 5.0}, 50.0, 0.90, 42);
    TEST_ASSERT(runnerRes.bestSolution.size() == 3, "SA runner dimension mismatch");
    return true;
}

bool test_hill_climbing() {
    ga::metaheuristics::HillClimbingConfig cfg;
    cfg.search.dimension = 3;
    cfg.search.iterations = 50;
    cfg.search.bounds = {-5.0, 5.0};
    cfg.search.seed = 42;
    cfg.stepSize = 0.05;
    cfg.restarts = 2;
    cfg.numNeighbors = 4;

    ga::metaheuristics::HillClimbingOptimizer hc(cfg);
    auto res = hc.optimize(sphere);

    TEST_ASSERT(res.bestSolution.size() == 3, "HC dimension mismatch");
    TEST_ASSERT(res.bestFitness > 0.0, "HC fitness must be positive");
    TEST_ASSERT(res.evaluations > 0, "HC evaluations counted");

    // Test runner
    auto runnerRes = ga::api::runHillClimbing(sphere, 3, 40, {-5.0, 5.0}, 0.05, 1, 42);
    TEST_ASSERT(runnerRes.bestSolution.size() == 3, "HC runner dimension mismatch");
    return true;
}

// ---------------- 3. Dynamic GA ----------------
bool test_dynamic_ga() {
    ga::adaptive::DynamicGAConfig cfg;
    cfg.search.dimension = 3;
    cfg.search.iterations = 25;
    cfg.search.populationSize = 30;
    cfg.search.bounds = {-5.0, 5.0};
    cfg.search.seed = 42;
    cfg.stagnationWindow = 4;
    cfg.strategy = ga::adaptive::DynamicStrategy::Hybrid;
    cfg.immigrantRatio = 0.2;
    cfg.detectEnvironmentChange = true;

    ga::adaptive::DynamicGAOptimizer dyn(cfg);
    auto res = dyn.optimize(sphere);

    TEST_ASSERT(res.bestSolution.size() == 3, "Dynamic GA dimension mismatch");
    TEST_ASSERT(res.bestFitness > 0.0, "Dynamic GA fitness must be positive");
    TEST_ASSERT(res.bestHistory.size() == 25, "Dynamic GA history size mismatch");

    // Test runner
    auto runnerRes = ga::api::runDynamicGa(sphere, 3, 15, {-5.0, 5.0}, 20, 42);
    TEST_ASSERT(runnerRes.bestSolution.size() == 3, "Dynamic GA runner dimension mismatch");
    return true;
}

// ---------------- 4. Constraint-Heavy Handling ----------------
bool test_deb_feasibility() {
    using namespace ga::constraints;

    // Rule 1: Feasible beats Infeasible
    DebProfile feas1{10.0, 0.0, true};
    DebProfile infeas1{100.0, 2.5, false};
    TEST_ASSERT(debIsBetter(feas1, infeas1), "Feasible must beat infeasible even if raw fitness is lower");
    TEST_ASSERT(!debIsBetter(infeas1, feas1), "Infeasible must not beat feasible");

    // Rule 2: Feasible vs Feasible -> better fitness wins
    DebProfile feas2{25.0, 0.0, true};
    TEST_ASSERT(debIsBetter(feas2, feas1), "Between feasible, higher fitness wins");
    TEST_ASSERT(!debIsBetter(feas1, feas2), "Lower fitness feasible loses");

    // Rule 3: Infeasible vs Infeasible -> lower violation wins
    DebProfile infeas2{500.0, 0.5, false}; // lower violation
    TEST_ASSERT(debIsBetter(infeas2, infeas1), "Between infeasible, lower violation wins");
    TEST_ASSERT(!debIsBetter(infeas1, infeas2), "Higher violation loses");

    return true;
}

bool test_adaptive_penalty() {
    using namespace ga::constraints;

    AdaptivePenaltyConfig pcfg;
    pcfg.baseCoefficient = 0.5;
    pcfg.alpha = 2.0;
    pcfg.beta = 2.0;

    AdaptivePenaltyHandler handler(pcfg);

    double pen0 = handler.penaltyFactor(0);
    double pen10 = handler.penaltyFactor(10);
    TEST_ASSERT(pen10 > pen0, "Penalty factor must increase with iterations");

    double raw = 100.0;
    double viol = 1.0;
    double penalized0 = handler.penalize(raw, viol, 0);
    double penalized10 = handler.penalize(raw, viol, 10);
    TEST_ASSERT(penalized0 < raw, "Penalized fitness must be less than raw for positive violation");
    TEST_ASSERT(penalized10 < penalized0, "Later generations must suffer stricter penalty");

    return true;
}

bool test_repair_operators() {
    using namespace ga::constraints;

    // Box bounds repair
    std::vector<double> genes = {-10.0, 2.0, 15.0};
    repairBoxBounds(genes, -5.0, 5.0);
    TEST_ASSERT(genes[0] == -5.0 && genes[1] == 2.0 && genes[2] == 5.0, "Box repair bounds clamped");

    // Simplex repair
    std::vector<double> weights = {0.2, 0.3, 0.5, 1.0};
    repairSimplex(weights, 1.0);
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    TEST_ASSERT(std::abs(sum - 1.0) < 1e-9, "Simplex repair sum must equal 1.0");

    // Permutation repair
    std::vector<int> perm = {0, 1, 1, 3}; // Duplicate 1, missing 2
    repairPermutation(perm);
    TEST_ASSERT(perm.size() == 4, "Permutation size preserved");
    std::vector<bool> seen(4, false);
    for (int p : perm) {
        TEST_ASSERT(p >= 0 && p < 4, "Permutation value in range");
        seen[p] = true;
    }
    for (bool s : seen) {
        TEST_ASSERT(s, "All permutation symbols present");
    }

    return true;
}

bool test_constrained_optimizer() {
    using namespace ga::constraints;

    ConstraintSet cset;
    // Constraint: sum of genes must be <= 1.0
    cset.hard.push_back([](const std::vector<double>& g) {
        double s = 0.0;
        for (double v : g) s += v;
        return s <= 1.0;
    });

    ConstrainedOptimizerConfig cfg;
    cfg.search.dimension = 3;
    cfg.search.iterations = 20;
    cfg.search.populationSize = 25;
    cfg.search.bounds = {-2.0, 2.0};
    cfg.search.seed = 42;
    cfg.useDebFeasibility = true;
    cfg.useAdaptivePenalty = true;

    ConstrainedOptimizer opt(cfg, cset);
    auto res = opt.optimize(sphere);

    TEST_ASSERT(res.bestSolution.size() == 3, "Constrained optimizer dimension mismatch");
    TEST_ASSERT(res.bestFitness > 0.0, "Constrained optimizer fitness positive");

    double solSum = std::accumulate(res.bestSolution.begin(), res.bestSolution.end(), 0.0);
    TEST_ASSERT(solSum <= 1.0001, "Best solution must satisfy hard constraint");

    // Test runner
    auto runnerRes = ga::api::runConstrainedOptimization(sphere, cset, 3, 15, {-2.0, 2.0}, 20, 42);
    TEST_ASSERT(runnerRes.bestSolution.size() == 3, "Constrained runner dimension mismatch");
    return true;
}

} // namespace

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " Alternatives & Advanced Sanity Tests   " << std::endl;
    std::cout << "========================================" << std::endl;

    RUN_TEST(test_island_model);
    RUN_TEST(test_simulated_annealing);
    RUN_TEST(test_hill_climbing);
    RUN_TEST(test_dynamic_ga);
    RUN_TEST(test_deb_feasibility);
    RUN_TEST(test_adaptive_penalty);
    RUN_TEST(test_repair_operators);
    RUN_TEST(test_constrained_optimizer);

    std::cout << "\nPassed: " << tests_passed << ", Failed: " << tests_failed << std::endl;
    return tests_failed == 0 ? 0 : 1;
}
