#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include "ga/metaheuristics.hpp"

namespace {

int failures = 0;

#define CHECK(condition, message)                                             \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::cerr << "[FAIL] " << (message) << '\n';                     \
            ++failures;                                                       \
            return;                                                           \
        }                                                                     \
    } while (false)

double sphereFitness(const std::vector<double>& solution) {
    double sum = 0.0;
    for (double value : solution) {
        sum += value * value;
    }
    return 1.0 / (1.0 + sum);
}

bool nonDecreasing(const std::vector<double>& values) {
    return std::adjacent_find(values.begin(), values.end(), std::greater<double>()) ==
           values.end();
}

bool nonIncreasing(const std::vector<double>& values) {
    return std::adjacent_find(values.begin(), values.end(), std::less<double>()) ==
           values.end();
}

void testPsoVariants() {
    const std::vector<ga::pso::PsoVariant> variants{
        ga::pso::PsoVariant::GlobalBest,
        ga::pso::PsoVariant::LocalBest,
        ga::pso::PsoVariant::Constriction,
        ga::pso::PsoVariant::BareBones,
        ga::pso::PsoVariant::FullyInformed,
        ga::pso::PsoVariant::QuantumBehaved,
    };

    for (ga::pso::PsoVariant variant : variants) {
        ga::pso::PsoConfig config;
        config.search.populationSize = 24;
        config.search.iterations = 35;
        config.search.dimension = 4;
        config.search.bounds = {-5.0, 5.0};
        config.search.seed = 71;
        config.search.threads = 2;
        config.variant = variant;
        ga::pso::ParticleSwarmOptimizer optimizer(config);
        const auto result = optimizer.optimize(sphereFitness);
        CHECK(result.bestSolution.size() == config.search.dimension,
              optimizer.name() + " returned a wrong-dimensional solution");
        CHECK(result.bestHistory.size() == config.search.iterations + 1,
              optimizer.name() + " returned incomplete history");
        CHECK(nonDecreasing(result.bestHistory),
              optimizer.name() + " best history is not monotonic");
        CHECK(result.evaluations ==
                  config.search.populationSize * (config.search.iterations + 1),
              optimizer.name() + " evaluation count is wrong");
        CHECK(result.bestFitness >= result.bestHistory.front(),
              optimizer.name() + " lost its initial best");
    }

    ga::pso::PsoConfig binaryConfig;
    binaryConfig.search.populationSize = 30;
    binaryConfig.search.iterations = 30;
    binaryConfig.search.dimension = 12;
    binaryConfig.search.bounds = {0.0, 1.0};
    binaryConfig.search.seed = 42;
    binaryConfig.variant = ga::pso::PsoVariant::Binary;
    ga::pso::ParticleSwarmOptimizer binary(binaryConfig);
    const auto result = binary.optimize([](const std::vector<double>& x) {
        return std::accumulate(x.begin(), x.end(), 0.0);
    });
    CHECK(result.bestFitness >= 9.0, "binary PSO did not optimize OneMax");
    CHECK(std::all_of(result.bestSolution.begin(), result.bestSolution.end(),
                      [](double x) { return x == 0.0 || x == 1.0; }),
          "binary PSO returned a non-binary solution");
    std::cout << "[PASS] PSO variants\n";
}

void testContinuousOptimizersAndFuzzyControl() {
    auto controller = std::make_shared<ga::fuzzy::FuzzyAdaptiveController>();

    ga::aco::AcorConfig acorConfig;
    acorConfig.search.iterations = 35;
    acorConfig.search.dimension = 4;
    acorConfig.search.bounds = {-5.0, 5.0};
    acorConfig.search.seed = 13;
    acorConfig.archiveSize = 30;
    acorConfig.sampleCount = 20;
    acorConfig.controller = controller;
    ga::aco::ContinuousAntColonyOptimizer acor(acorConfig);
    const auto acorResult = acor.optimize(sphereFitness, {{4.0, 4.0, 4.0, 4.0}});
    CHECK(acorResult.bestSolution.size() == 4, "ACOR result dimension is wrong");
    CHECK(nonDecreasing(acorResult.bestHistory), "ACOR history is not monotonic");
    CHECK(acorResult.evaluations == 30 + 35 * 20, "ACOR evaluation count is wrong");

    ga::gsa::GsaConfig gsaConfig;
    gsaConfig.search.populationSize = 30;
    gsaConfig.search.iterations = 35;
    gsaConfig.search.dimension = 4;
    gsaConfig.search.bounds = {-5.0, 5.0};
    gsaConfig.search.seed = 19;
    gsaConfig.controller = controller;
    ga::gsa::GravitationalSearchOptimizer gsa(gsaConfig);
    const auto gsaResult = gsa.optimize(sphereFitness);
    CHECK(gsaResult.bestSolution.size() == 4, "GSA result dimension is wrong");
    CHECK(nonDecreasing(gsaResult.bestHistory), "GSA history is not monotonic");

    ga::metaheuristics::ProgressState stuck;
    stuck.normalizedDiversity = 0.02;
    stuck.relativeImprovement = 0.0;
    stuck.stagnation = 1.0;
    const auto explore = controller->update(stuck);
    CHECK(explore.exploration > 1.0 && explore.randomization > 1.0,
          "fuzzy controller did not increase exploration when stuck");

    ga::fuzzy::FuzzyControllerConfig userFuzzyConfig;
    userFuzzyConfig.lowDiversityStagnant = {12.25, 0.70, 1.40, 1.80};
    userFuzzyConfig.lowDiversitySlow = userFuzzyConfig.lowDiversityStagnant;
    const ga::fuzzy::FuzzyAdaptiveController userController(userFuzzyConfig);
    ga::metaheuristics::ProgressState userState = stuck;
    userState.normalizedDiversity = 0.0;
    const auto userSignal = userController.update(userState);
    CHECK(std::abs(userSignal.exploration - 12.25) < 1e-12 &&
              std::abs(userSignal.exploitation - 0.70) < 1e-12 &&
              std::abs(userSignal.evaporation - 1.40) < 1e-12 &&
              std::abs(userSignal.randomization - 1.80) < 1e-12,
          "fuzzy controller did not honor user-defined consequents");
    const auto forwarded = ga::metaheuristics::detail::controlSignal(
        &userController, 0, 1, 0.0, 0.0, 1);
    CHECK(std::abs(forwarded.exploration - 12.25) < 1e-12,
          "shared controller path changed a valid user-defined multiplier");

    bool invalidRejected = false;
    try {
        ga::fuzzy::FuzzyControllerConfig invalidConfig;
        invalidConfig.improvementScale = 0.0;
        const ga::fuzzy::FuzzyAdaptiveController invalidController(invalidConfig);
        (void)invalidController;
    } catch (const std::invalid_argument&) {
        invalidRejected = true;
    }
    CHECK(invalidRejected, "fuzzy controller accepted an invalid user configuration");
    std::cout << "[PASS] ACOR, GSA, and fuzzy control\n";
}

void testGraphAcoVariants() {
    ga::aco::DenseGraph graph({
        {0, 2, 9, 10, 7},
        {2, 0, 6, 4, 3},
        {9, 6, 0, 8, 5},
        {10, 4, 8, 0, 6},
        {7, 3, 5, 6, 0},
    });
    const std::vector<ga::aco::AntColonyVariant> variants{
        ga::aco::AntColonyVariant::AntSystem,
        ga::aco::AntColonyVariant::ElitistAntSystem,
        ga::aco::AntColonyVariant::RankBasedAntSystem,
        ga::aco::AntColonyVariant::AntColonySystem,
        ga::aco::AntColonyVariant::MaxMinAntSystem,
    };

    for (ga::aco::AntColonyVariant variant : variants) {
        ga::aco::AntColonyConfig config;
        config.ants = 20;
        config.iterations = 25;
        config.seed = 23;
        config.variant = variant;
        config.controller = std::make_shared<ga::fuzzy::FuzzyAdaptiveController>();
        const auto result = ga::aco::AntColonyOptimizer(config).solve(graph);
        CHECK(result.bestTour.size() == graph.size(), "ACO returned an incomplete tour");
        CHECK(std::set<std::size_t>(result.bestTour.begin(), result.bestTour.end()).size() ==
                  graph.size(),
              "ACO returned a tour with duplicate nodes");
        CHECK(std::isfinite(result.bestCost) && result.bestCost > 0.0,
              "ACO returned an invalid tour cost");
        CHECK(result.bestCostHistory.size() == config.iterations,
              "ACO returned incomplete history");
        CHECK(nonIncreasing(result.bestCostHistory), "ACO best-cost history is not monotonic");
        CHECK(result.evaluations == config.ants * config.iterations,
              "ACO evaluation count is wrong");
    }
    std::cout << "[PASS] graph ACO variants\n";
}

void testFuzzyCMeans() {
    const std::vector<std::vector<double>> data{
        {-0.1, 0.0}, {0.0, 0.1}, {0.1, -0.1},
        {9.9, 10.0}, {10.0, 10.1}, {10.1, 9.9},
    };
    ga::fuzzy::FuzzyCMeansConfig config;
    config.clusters = 2;
    config.seed = 42;
    config.tolerance = 1e-8;
    const auto result = ga::fuzzy::FuzzyCMeans(config).fit(data);
    CHECK(result.centers.size() == 2, "FCM returned the wrong number of centers");
    CHECK(result.membership.size() == data.size(), "FCM returned incomplete membership");
    CHECK(result.converged, "FCM did not converge on separated clusters");
    CHECK(nonIncreasing(result.objectiveHistory), "FCM objective history is not monotonic");
    for (const auto& row : result.membership) {
        const double sum = std::accumulate(row.begin(), row.end(), 0.0);
        CHECK(std::abs(sum - 1.0) < 1e-9, "FCM memberships do not sum to one");
    }
    const auto labels = result.labels();
    CHECK(labels[0] == labels[1] && labels[1] == labels[2],
          "FCM split the first cluster");
    CHECK(labels[3] == labels[4] && labels[4] == labels[5] && labels[0] != labels[3],
          "FCM failed to separate the second cluster");
    std::cout << "[PASS] fuzzy C-means\n";
}

void testHybridAndGaEvaluationEfficiency() {
    ga::Config gaConfig;
    gaConfig.populationSize = 11;
    gaConfig.generations = 8;
    gaConfig.dimension = 3;
    gaConfig.bounds = {-5.0, 5.0};
    gaConfig.eliteRatio = 0.0;
    gaConfig.seed = 7;

    std::atomic<std::size_t> calls{0};
    ga::GeneticAlgorithm gaOnly(gaConfig);
    const auto gaResult = gaOnly.run(
        [&](const std::vector<double>& x) {
            ++calls;
            return sphereFitness(x);
        },
        {{0.0, 0.0, 0.0}});
    CHECK(calls == gaResult.evaluations, "GA evaluation accounting is wrong");
    CHECK(std::abs(gaResult.bestFitness - 1.0) < 1e-12,
          "GA forgot a global best when elitism was disabled");
    const std::size_t elites = static_cast<std::size_t>(
        std::round(gaConfig.eliteRatio * gaConfig.populationSize));
    const std::size_t expectedCalls = static_cast<std::size_t>(gaConfig.populationSize) +
        static_cast<std::size_t>(gaConfig.generations) *
            (static_cast<std::size_t>(gaConfig.populationSize) - elites);
    CHECK(calls == expectedCalls, "GA still performs redundant offspring evaluations");

    ga::pso::PsoConfig psoConfig;
    psoConfig.search.populationSize = 20;
    psoConfig.search.iterations = 20;
    psoConfig.search.dimension = 3;
    psoConfig.search.bounds = {-5.0, 5.0};
    psoConfig.search.seed = 8;

    ga::hybrid::MetaheuristicPipeline pipeline;
    pipeline.add(std::make_unique<ga::metaheuristics::GeneticAlgorithmAdapter>(gaConfig))
        .add(std::make_unique<ga::pso::ParticleSwarmOptimizer>(psoConfig));
    const auto result = pipeline.optimizeDetailed(sphereFitness);
    CHECK(result.stages.size() == 2, "hybrid pipeline did not run every stage");
    CHECK(result.stages[0].optimizer == "GA" &&
              result.stages[1].optimizer == "PSO-global-best",
          "hybrid pipeline did not preserve the user-selected stage order");
    CHECK(result.combined.bestFitness >= result.stages[0].result.bestFitness,
          "hybrid pipeline lost the GA solution");
    CHECK(result.combined.evaluations == result.stages[0].result.evaluations +
                                               result.stages[1].result.evaluations,
          "hybrid pipeline evaluation accounting is wrong");
    std::cout << "[PASS] hybrid pipeline and GA evaluation efficiency\n";
}

} // namespace

int main() {
    testPsoVariants();
    testContinuousOptimizersAndFuzzyControl();
    testGraphAcoVariants();
    testFuzzyCMeans();
    testHybridAndGaEvaluationEfficiency();

    if (failures != 0) {
        std::cerr << failures << " metaheuristic test group(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All metaheuristic tests passed\n";
    return EXIT_SUCCESS;
}
