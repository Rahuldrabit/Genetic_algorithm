#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
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

void testFuzzyMembershipAndOperators() {
    // 1. Triangular MF
    auto tri = ga::fuzzy::makeTriangularMF(0.0, 5.0, 10.0);
    CHECK(std::abs(tri->evaluate(0.0) - 0.0) < 1e-9, "Triangular left boundary failed");
    CHECK(std::abs(tri->evaluate(5.0) - 1.0) < 1e-9, "Triangular peak failed");
    CHECK(std::abs(tri->evaluate(2.5) - 0.5) < 1e-9, "Triangular midpoint failed");
    CHECK(std::abs(tri->evaluate(10.0) - 0.0) < 1e-9, "Triangular right boundary failed");
    CHECK(std::abs(tri->evaluate(12.0) - 0.0) < 1e-9, "Triangular out of bounds failed");

    // 2. Trapezoidal MF
    auto trap = ga::fuzzy::makeTrapezoidalMF(1.0, 3.0, 7.0, 9.0);
    CHECK(std::abs(trap->evaluate(1.0) - 0.0) < 1e-9, "Trapezoidal a failed");
    CHECK(std::abs(trap->evaluate(3.0) - 1.0) < 1e-9, "Trapezoidal b failed");
    CHECK(std::abs(trap->evaluate(5.0) - 1.0) < 1e-9, "Trapezoidal plateau failed");
    CHECK(std::abs(trap->evaluate(7.0) - 1.0) < 1e-9, "Trapezoidal c failed");
    CHECK(std::abs(trap->evaluate(9.0) - 0.0) < 1e-9, "Trapezoidal d failed");

    // 3. Gaussian MF
    auto gauss = ga::fuzzy::makeGaussianMF(5.0, 1.0);
    CHECK(std::abs(gauss->evaluate(5.0) - 1.0) < 1e-9, "Gaussian center failed");
    CHECK(gauss->evaluate(6.0) < 1.0 && gauss->evaluate(6.0) > 0.0, "Gaussian decay failed");

    // 4. T-Norms (Fuzzy AND)
    CHECK(std::abs(ga::fuzzy::applyTNorm(ga::fuzzy::TNormType::Minimum, 0.4, 0.7) - 0.4) < 1e-9, "TNorm Minimum failed");
    CHECK(std::abs(ga::fuzzy::applyTNorm(ga::fuzzy::TNormType::AlgebraicProduct, 0.4, 0.5) - 0.2) < 1e-9, "TNorm AlgebraicProduct failed");
    CHECK(std::abs(ga::fuzzy::applyTNorm(ga::fuzzy::TNormType::Lukasiewicz, 0.6, 0.7) - 0.3) < 1e-9, "TNorm Lukasiewicz failed");

    // 5. S-Norms (Fuzzy OR)
    CHECK(std::abs(ga::fuzzy::applySNorm(ga::fuzzy::SNormType::Maximum, 0.4, 0.7) - 0.7) < 1e-9, "SNorm Maximum failed");
    CHECK(std::abs(ga::fuzzy::applySNorm(ga::fuzzy::SNormType::AlgebraicSum, 0.4, 0.5) - 0.7) < 1e-9, "SNorm AlgebraicSum failed");
    CHECK(std::abs(ga::fuzzy::applySNorm(ga::fuzzy::SNormType::Lukasiewicz, 0.6, 0.7) - 1.0) < 1e-9, "SNorm Lukasiewicz failed");

    // 6. Complements (Fuzzy NOT)
    CHECK(std::abs(ga::fuzzy::applyComplement(ga::fuzzy::ComplementType::Standard, 0.3) - 0.7) < 1e-9, "Complement Standard failed");

    // 7. Defuzzification (Centroid of symmetric triangle centered at 5.0)
    double centroid = ga::fuzzy::defuzzify(
        [&tri](double x) { return tri->evaluate(x); },
        {0.0, 10.0},
        ga::fuzzy::DefuzzMethod::Centroid);
    CHECK(std::abs(centroid - 5.0) < 0.1, "Centroid defuzzification of symmetric triangle failed");

    std::cout << "[PASS] Fuzzy Membership, Operators & Defuzzification\n";
}

void testMamdaniAndSugenoInference() {
    // 1. Mamdani System
    ga::fuzzy::MamdaniSystem mamdani;
    mamdani.addInput("Service", 0.0, 10.0);
    mamdani.addInputTerm("Service", "Poor", ga::fuzzy::makeTriangularMF(-1.0, 0.0, 5.0));
    mamdani.addInputTerm("Service", "Good", ga::fuzzy::makeTriangularMF(0.0, 5.0, 10.0));
    mamdani.addInputTerm("Service", "Great", ga::fuzzy::makeTriangularMF(5.0, 10.0, 11.0));

    mamdani.addOutput("Tip", 0.0, 30.0);
    mamdani.addOutputTerm("Tip", "Low", ga::fuzzy::makeTriangularMF(0.0, 5.0, 15.0));
    mamdani.addOutputTerm("Tip", "Med", ga::fuzzy::makeTriangularMF(10.0, 15.0, 20.0));
    mamdani.addOutputTerm("Tip", "High", ga::fuzzy::makeTriangularMF(15.0, 25.0, 30.0));

    mamdani.addRule("IF Service IS Poor THEN Tip IS Low");
    mamdani.addRule("IF Service IS Good THEN Tip IS Med");
    mamdani.addRule("IF Service IS Great THEN Tip IS High");

    double lowTip = mamdani.evaluateSingle("Tip", {{"Service", 1.0}});
    double highTip = mamdani.evaluateSingle("Tip", {{"Service", 9.0}});
    CHECK(lowTip < 12.0, "Mamdani low tip failed");
    CHECK(highTip > 18.0, "Mamdani high tip failed");
    CHECK(highTip > lowTip, "Mamdani monotonicity failed");

    // 2. Sugeno System
    ga::fuzzy::SugenoSystem sugeno;
    sugeno.addInput("Input", 0.0, 10.0);
    sugeno.addInputTerm("Input", "Low", ga::fuzzy::makeTriangularMF(-1.0, 0.0, 5.0));
    sugeno.addInputTerm("Input", "High", ga::fuzzy::makeTriangularMF(5.0, 10.0, 11.0));
    sugeno.addOutput("Output", 0.0);

    // Rule 1: IF Input IS Low THEN Output = 2.0
    // Rule 2: IF Input IS High THEN Output = 10.0
    sugeno.addRule(ga::fuzzy::SugenoRule(
        {{"Input", "Low", false}},
        {{"Output", 2.0, {}}}
    ));
    sugeno.addRule(ga::fuzzy::SugenoRule(
        {{"Input", "High", false}},
        {{"Output", 10.0, {}}}
    ));

    double sugLow = sugeno.evaluateSingle("Output", {{"Input", 0.0}});
    double sugHigh = sugeno.evaluateSingle("Output", {{"Input", 10.0}});
    CHECK(std::abs(sugLow - 2.0) < 1e-9, "Sugeno low output failed");
    CHECK(std::abs(sugHigh - 10.0) < 1e-9, "Sugeno high output failed");

    std::cout << "[PASS] Mamdani and Sugeno Inference Engines\n";
}

void testFcmAndFuzzyAdapter() {
    // 1. FCM Swarm Analyzer
    ga::fuzzy::FcmSwarmAnalyzer analyzer;
    std::vector<std::vector<double>> population = {
        {0.0, 0.0}, {0.1, 0.1}, {0.05, -0.05},
        {5.0, 5.0}, {5.1, 4.9}, {4.9, 5.1},
        {10.0, 10.0}, {10.1, 9.9}, {9.9, 10.1}
    };
    auto metrics = analyzer.analyze(population);
    CHECK(metrics.clusterCenters.size() == 3, "FCM analyzer cluster count mismatch");
    CHECK(metrics.partitionCoefficient > 0.33, "FCM partition coefficient invalid");
    CHECK(std::isfinite(metrics.compactness), "FCM compactness not finite");

    // 2. FuzzyAlgorithmAdapter
    ga::fuzzy::FuzzyAlgorithmAdapter adapter;
    ga::metaheuristics::ProgressState state;
    state.normalizedDiversity = 0.05;
    state.stagnation = 0.9;
    state.relativeImprovement = 0.0;

    auto signal = adapter.updateWithFcm(state, metrics);
    CHECK(signal.exploration > 1.0, "Fuzzy adapter should increase exploration on stagnation");
    CHECK(adapter.lastShakeIntensity() > 0.0, "Fuzzy adapter should trigger shake on high stagnation");

    std::cout << "[PASS] FCM Swarm Analyzer and Fuzzy Algorithm Adapter\n";
}

void testFuzzyAdaptiveCrossover() {
    ga::fuzzy::FuzzyAdaptiveCrossover xover;
    RealVector p1 = {1.0, 2.0, 3.0, 4.0};
    RealVector p2 = {1.1, 2.1, 2.9, 4.1};

    xover.updateContext(/*diversity=*/0.1, /*progress=*/0.8);
    auto [c1, c2] = xover.crossover(p1, p2);

    CHECK(c1.size() == p1.size(), "Child 1 dimension mismatch");
    CHECK(c2.size() == p2.size(), "Child 2 dimension mismatch");
    CHECK(xover.getOperationCount() == 1, "Operation count tracking failed");

    std::cout << "[PASS] Fuzzy Adaptive Crossover\n";
}

void testShakeOperatorsAndAdaptiveShaker() {
    std::mt19937 rng(42);
    ga::Bounds bounds{-5.0, 5.0};
    std::vector<double> original = {0.0, 0.0, 0.0, 0.0};

    // Cauchy Shake
    std::vector<double> cauchySol = original;
    ga::shake::cauchyShake(cauchySol, bounds, 0.1, rng);
    CHECK(cauchySol != original, "Cauchy shake failed to perturb");
    for (double x : cauchySol) {
        CHECK(x >= bounds.lower && x <= bounds.upper, "Cauchy shake exceeded bounds");
    }

    // Levy Flight Shake
    std::vector<double> levySol = original;
    ga::shake::levyFlightShake(levySol, bounds, 0.1, 1.5, rng);
    CHECK(levySol != original, "Levy flight shake failed to perturb");
    for (double x : levySol) {
        CHECK(x >= bounds.lower && x <= bounds.upper, "Levy flight shake exceeded bounds");
    }

    // Opposition Shake
    std::vector<double> oppSol = {1.0, -2.0, 3.0, -4.0};
    ga::shake::oppositionShake(oppSol, bounds);
    CHECK(std::abs(oppSol[0] - (-1.0)) < 1e-9, "Opposition shake dim 0 failed");
    CHECK(std::abs(oppSol[1] - (2.0)) < 1e-9, "Opposition shake dim 1 failed");

    // Adaptive Swarm Shaker
    ga::shake::AdaptiveShakerConfig shakerCfg;
    shakerCfg.stagnationThreshold = 2;
    shakerCfg.diversityThreshold = 0.5;
    ga::shake::AdaptiveSwarmShaker shaker(shakerCfg);

    std::vector<std::vector<double>> pop = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    std::vector<double> fits = {1.0, 1.0, 1.0};
    // Iteration 1: initial
    shaker.checkAndShake(pop, fits, 0, bounds, sphereFitness, rng);
    // Iteration 2: stagnant
    shaker.checkAndShake(pop, fits, 0, bounds, sphereFitness, rng);
    // Iteration 3: triggers shake!
    bool shaken = shaker.checkAndShake(pop, fits, 0, bounds, sphereFitness, rng);
    CHECK(shaken, "AdaptiveSwarmShaker failed to trigger on stagnation");

    std::cout << "[PASS] Shake Operators and Adaptive Swarm Shaker\n";
}

void testGlowwormSwarmOptimization() {
    const std::vector<ga::gso::GsoVariant> variants{
        ga::gso::GsoVariant::Standard,
        ga::gso::GsoVariant::AdaptiveStep,
        ga::gso::GsoVariant::LevyFlight,
        ga::gso::GsoVariant::MultiModal
    };

    for (auto variant : variants) {
        ga::gso::GsoConfig cfg;
        cfg.search.populationSize = 25;
        cfg.search.iterations = 30;
        cfg.search.dimension = 4;
        cfg.search.bounds = {-5.0, 5.0};
        cfg.search.seed = 99;
        cfg.variant = variant;

        ga::gso::GlowwormSwarmOptimizer gso(cfg);
        auto res = gso.optimize(sphereFitness);

        CHECK(res.bestSolution.size() == 4, "GSO result dimension mismatch");
        CHECK(res.bestFitness > 0.0, "GSO invalid fitness");
        CHECK(res.bestHistory.size() == cfg.search.iterations + 1, "GSO history size mismatch");
        CHECK(res.evaluations == cfg.search.populationSize * (cfg.search.iterations + 1), "GSO evaluations count mismatch");
    }

    std::cout << "[PASS] Glowworm Swarm Optimization (All Variants)\n";
}

void testPsoExtendedVariants() {
    // 1. CLPSO standalone optimizer
    ga::pso::ClpsoConfig clCfg;
    clCfg.search.populationSize = 24;
    clCfg.search.iterations = 30;
    clCfg.search.dimension = 4;
    clCfg.search.bounds = {-5.0, 5.0};
    clCfg.search.seed = 77;
    ga::pso::ComprehensiveLearningPso clpso(clCfg);
    auto resCl = clpso.optimize(sphereFitness);
    CHECK(resCl.bestSolution.size() == 4, "CLPSO dimension mismatch");
    CHECK(resCl.bestFitness > 0.0, "CLPSO invalid fitness");

    // 2. Baseline PSO variants
    const std::vector<ga::pso::PsoVariant> variants{
        ga::pso::PsoVariant::GlobalBest,
        ga::pso::PsoVariant::LocalBest,
        ga::pso::PsoVariant::Constriction,
        ga::pso::PsoVariant::BareBones,
        ga::pso::PsoVariant::FullyInformed,
        ga::pso::PsoVariant::QuantumBehaved
    };

    for (auto variant : variants) {
        ga::pso::PsoConfig cfg;
        cfg.search.populationSize = 24;
        cfg.search.iterations = 30;
        cfg.search.dimension = 4;
        cfg.search.bounds = {-5.0, 5.0};
        cfg.search.seed = 77;
        cfg.variant = variant;

        ga::pso::ParticleSwarmOptimizer pso(cfg);
        auto res = pso.optimize(sphereFitness);

        CHECK(res.bestSolution.size() == 4, "PSO variant dimension mismatch");
        CHECK(res.bestFitness > 0.0, "PSO variant invalid fitness");
    }

    std::cout << "[PASS] Extended PSO Variants (CLPSO and Standard Variants)\n";
}

void testContinuousAcoExtendedVariants() {
    ga::aco::AcorConfig cfg;
    cfg.search.iterations = 25;
    cfg.search.dimension = 4;
    cfg.search.bounds = {-5.0, 5.0};
    cfg.search.seed = 55;
    cfg.archiveSize = 25;
    cfg.sampleCount = 15;

    ga::aco::ContinuousAntColonyOptimizer acor(cfg);
    auto res = acor.optimize(sphereFitness);

    CHECK(res.bestSolution.size() == 4, "ACOR dimension mismatch");
    CHECK(res.bestFitness > 0.0, "ACOR invalid fitness");
    CHECK(res.evaluations == cfg.archiveSize + cfg.search.iterations * cfg.sampleCount, "ACOR evaluation count mismatch");

    std::cout << "[PASS] Continuous ACO (ACOR)\n";
}


void testCustomOptimizerAndHybrids() {
    // 1. Open Own Algorithm / CustomContinuousOptimizer
    auto custom = ga::metaheuristics::makeCustomOptimizer(
        "CustomLocalSearch",
        ga::metaheuristics::SearchConfig{10, 20, 3, {-5.0, 5.0}, 123},
        [](const ga::Fitness& fitness,
           const ga::metaheuristics::SearchConfig& cfg,
           const ga::metaheuristics::SeedPopulation& seeds,
           const ga::metaheuristics::IAdaptiveController*) {
            ga::core::OptimizationResult res;
            std::vector<double> cur = seeds.empty() ? std::vector<double>(cfg.dimension, 1.0) : seeds.front();
            double fit = fitness(cur);
            res.bestSolution = cur;
            res.bestFitness = fit;
            res.evaluations = 1;
            return res;
        });

    auto custRes = custom->optimize(sphereFitness);
    CHECK(custRes.bestSolution.size() == 3, "Custom optimizer dimension mismatch");

    // 2. Interleaved Co-Evolutionary Hybrid (PSO <-> GSO)
    ga::pso::PsoConfig psoCfg;
    psoCfg.search.populationSize = 15;
    psoCfg.search.iterations = 10;
    psoCfg.search.dimension = 3;
    psoCfg.search.bounds = {-5.0, 5.0};
    psoCfg.search.seed = 11;
    auto pso = std::make_shared<ga::pso::ParticleSwarmOptimizer>(psoCfg);

    ga::gso::GsoConfig gsoCfg;
    gsoCfg.search = psoCfg.search;
    gsoCfg.search.seed = 22;
    auto gso = std::make_shared<ga::gso::GlowwormSwarmOptimizer>(gsoCfg);

    ga::hybrid::InterleavedHybridConfig interCfg;
    interCfg.search = psoCfg.search;
    interCfg.epochs = 2;
    ga::hybrid::InterleavedHybridOptimizer interHybrid(pso, gso, interCfg);

    auto interRes = interHybrid.optimize(sphereFitness);
    CHECK(interRes.bestSolution.size() == 3, "Interleaved hybrid dimension mismatch");
    CHECK(interRes.bestFitness > 0.0, "Interleaved hybrid invalid fitness");

    // 3. Swarm-Genetic Hybrid
    ga::hybrid::SwarmGeneticHybridConfig sgCfg;
    sgCfg.search = psoCfg.search;
    sgCfg.search.iterations = 20;
    ga::hybrid::SwarmGeneticHybridOptimizer sgHybrid(sgCfg);
    auto sgRes = sgHybrid.optimize(sphereFitness);
    CHECK(sgRes.bestSolution.size() == 3, "Swarm-Genetic hybrid dimension mismatch");
    CHECK(sgRes.bestFitness > 0.0, "Swarm-Genetic hybrid invalid fitness");

    std::cout << "[PASS] Custom Optimizer and Hybrid Systems\n";
}

void testStandaloneRunnersAndDispatcher() {
    // 1. runPso & runClpso
    auto resPso = ga::api::runPso(sphereFitness, ga::pso::PsoVariant::GlobalBest, 4, 20);
    CHECK(resPso.bestSolution.size() == 4, "runPso dimension mismatch");

    auto resCl = ga::api::runClpso(sphereFitness, 4, 20);
    CHECK(resCl.bestSolution.size() == 4, "runClpso dimension mismatch");

    // 2. runContinuousAco
    auto resAcor = ga::api::runContinuousAco(sphereFitness, 4, 20);
    CHECK(resAcor.bestSolution.size() == 4, "runContinuousAco dimension mismatch");

    // 3. runGso
    auto resGso = ga::api::runGso(sphereFitness, ga::gso::GsoVariant::AdaptiveStep, 4, 20);
    CHECK(resGso.bestSolution.size() == 4, "runGso dimension mismatch");

    // 4. runGsa
    auto resGsa = ga::api::runGsa(sphereFitness, 4, 20);
    CHECK(resGsa.bestSolution.size() == 4, "runGsa dimension mismatch");

    // 5. runGeneticAlgorithm
    auto resGa = ga::api::runGeneticAlgorithm(sphereFitness, 4, 20);
    CHECK(resGa.bestSolution.size() == 4, "runGeneticAlgorithm dimension mismatch");

    // 6. solve dispatcher
    ga::metaheuristics::SearchConfig cfg;
    cfg.dimension = 4;
    cfg.iterations = 20;
    cfg.bounds = {-5.0, 5.0};
    auto resSolve = ga::api::solve(ga::api::AlgorithmType::ComprehensiveLearningPso, sphereFitness, cfg);
    CHECK(resSolve.bestSolution.size() == 4, "solve dispatcher dimension mismatch");

    std::cout << "[PASS] Standalone Runners and Unified Solver Dispatcher\n";
}

} // namespace

int main() {
    testFuzzyMembershipAndOperators();
    testMamdaniAndSugenoInference();
    testFcmAndFuzzyAdapter();
    testFuzzyAdaptiveCrossover();
    testShakeOperatorsAndAdaptiveShaker();
    testGlowwormSwarmOptimization();
    testPsoExtendedVariants();
    testContinuousAcoExtendedVariants();
    testCustomOptimizerAndHybrids();
    testStandaloneRunnersAndDispatcher();

    if (failures != 0) {
        std::cerr << failures << " extended test group(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All extended metaheuristic & fuzzy tests passed successfully!\n";
    return EXIT_SUCCESS;
}
