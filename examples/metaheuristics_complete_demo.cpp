#include <iostream>
#include <memory>
#include <vector>

#include "ga/metaheuristics.hpp"

namespace {

// Objective function: Sphere (higher fitness = closer to origin)
double sphereFitness(const std::vector<double>& x) {
    double sum = 0.0;
    for (double val : x) sum += val * val;
    return 1.0 / (1.0 + sum);
}

// Rastrigin multimodal function
double rastriginFitness(const std::vector<double>& x) {
    const double pi = 3.14159265358979323846;
    double sum = 10.0 * static_cast<double>(x.size());
    for (double val : x) {
        sum += val * val - 10.0 * std::cos(2.0 * pi * val);
    }
    return 1.0 / (1.0 + sum);
}

} // namespace

int main() {
    std::cout << "========================================================\n";
    std::cout << "  Metaheuristics, Fuzzy Systems & Hybrids Complete Demo\n";
    std::cout << "========================================================\n\n";

    // -------------------------------------------------------------------------
    // 1. Standalone Runners: Call Algorithms Separately in One Line
    // -------------------------------------------------------------------------
    std::cout << "--- 1. Standalone Single-Line Runners ---\n";

    // Run Comprehensive Learning PSO
    auto clpsoRes = ga::api::runClpso(sphereFitness,
                                      /*dimension=*/6, /*iterations=*/40,
                                      /*bounds=*/{-5.12, 5.12}, /*seed=*/42);
    std::cout << "CLPSO Best Fitness: " << clpsoRes.bestFitness
              << " (Evaluations: " << clpsoRes.evaluations << ")\n";

    // Run Glowworm Swarm Optimization with Adaptive Step Size
    auto gsoRes = ga::api::runGso(sphereFitness,
                                  ga::gso::GsoVariant::AdaptiveStep,
                                  /*dimension=*/6, /*iterations=*/40,
                                  /*bounds=*/{-5.12, 5.12}, /*seed=*/43);
    std::cout << "GSO-AdaptiveStep Best Fitness: " << gsoRes.bestFitness
              << " (Evaluations: " << gsoRes.evaluations << ")\n";

    // Run Continuous ACO
    auto acorRes = ga::api::runContinuousAco(sphereFitness,
                                             /*dimension=*/6, /*iterations=*/40,
                                             /*bounds=*/{-5.12, 5.12}, /*seed=*/44);
    std::cout << "ACOR Best Fitness: " << acorRes.bestFitness
              << " (Evaluations: " << acorRes.evaluations << ")\n";

    // Run Graph ACO for TSP
    ga::aco::DenseGraph tspGraph({
        {0, 2, 9, 10, 7},
        {2, 0, 6, 4, 3},
        {9, 6, 0, 8, 5},
        {10, 4, 8, 0, 6},
        {7, 3, 5, 6, 0}
    });
    auto tspRes = ga::api::runGraphAco(tspGraph, ga::aco::AntColonyVariant::MaxMinAntSystem);
    std::cout << "Graph ACO (MMAS) Best TSP Tour Cost: " << tspRes.bestCost << "\n\n";

    // -------------------------------------------------------------------------
    // 2. Open Own Algorithm with Cauchy Shake Perturbation
    // -------------------------------------------------------------------------
    std::cout << "--- 2. Open Own Algorithm with Cauchy Shake ---\n";
    auto myCustomAlgo = ga::metaheuristics::makeCustomOptimizer(
        "CauchyRandomWalk",
        ga::metaheuristics::SearchConfig{/*popSize=*/20, /*iters=*/30, /*dim=*/6, {-5.0, 5.0}, 101},
        [](const ga::Fitness& fitness,
           const ga::metaheuristics::SearchConfig& cfg,
           const ga::metaheuristics::SeedPopulation& seeds,
           const ga::metaheuristics::IAdaptiveController*) {
            
            std::mt19937 rng(cfg.seed == 0 ? std::random_device{}() : cfg.seed);
            std::vector<double> cur = seeds.empty() ? std::vector<double>(cfg.dimension, 1.0) : seeds.front();
            double curFit = fitness(cur);

            ga::core::OptimizationResult res;
            res.bestSolution = cur;
            res.bestFitness = curFit;

            for (std::size_t it = 0; it < cfg.iterations; ++it) {
                std::vector<double> candidate = cur;
                // Apply heavy-tailed Cauchy shake to jump across basins
                ga::shake::cauchyShake(candidate, cfg.bounds, 0.05, rng);
                double candFit = fitness(candidate);
                if (candFit > curFit) {
                    cur = candidate;
                    curFit = candFit;
                    if (curFit > res.bestFitness) {
                        res.bestFitness = curFit;
                        res.bestSolution = cur;
                    }
                }
                res.bestHistory.push_back(res.bestFitness);
            }
            res.evaluations = cfg.iterations;
            return res;
        });

    auto customRes = myCustomAlgo->optimize(sphereFitness);
    std::cout << "Custom Algorithm Best Fitness: " << customRes.bestFitness << "\n\n";

    // -------------------------------------------------------------------------
    // 3. Full Mamdani Fuzzy System Controlling PSO Online
    // -------------------------------------------------------------------------
    std::cout << "--- 3. Mamdani Fuzzy System Adapting PSO Online ---\n";
    ga::fuzzy::MamdaniSystem fis;
    fis.addInput("Diversity", 0.0, 1.0);
    fis.addInputTerm("Diversity", "Low",  ga::fuzzy::makeTriangularMF(-0.2, 0.0, 0.45));
    fis.addInputTerm("Diversity", "Mid",  ga::fuzzy::makeTriangularMF(0.25, 0.50, 0.75));
    fis.addInputTerm("Diversity", "High", ga::fuzzy::makeTriangularMF(0.55, 1.0, 1.2));

    fis.addInput("Stagnation", 0.0, 1.0);
    fis.addInputTerm("Stagnation", "Low",  ga::fuzzy::makeTriangularMF(-0.2, 0.0, 0.35));
    fis.addInputTerm("Stagnation", "High", ga::fuzzy::makeTriangularMF(0.25, 1.0, 1.2));

    fis.addOutput("Exploration", 0.5, 2.0);
    fis.addOutputTerm("Exploration", "Low",  ga::fuzzy::makeTriangularMF(0.5, 0.75, 1.0));
    fis.addOutputTerm("Exploration", "Norm", ga::fuzzy::makeTriangularMF(0.85, 1.0, 1.15));
    fis.addOutputTerm("Exploration", "High", ga::fuzzy::makeTriangularMF(1.0, 1.6, 2.0));

    fis.addOutput("Exploitation", 0.5, 2.0);
    fis.addOutputTerm("Exploitation", "Low",  ga::fuzzy::makeTriangularMF(0.5, 0.75, 1.0));
    fis.addOutputTerm("Exploitation", "High", ga::fuzzy::makeTriangularMF(1.0, 1.5, 2.0));

    fis.setTNorm(ga::fuzzy::TNormType::Minimum);
    fis.setSNorm(ga::fuzzy::SNormType::Maximum);
    fis.setDefuzzification(ga::fuzzy::DefuzzMethod::Centroid);

    fis.addRule("IF Diversity IS Low AND Stagnation IS High THEN Exploration IS High AND Exploitation IS Low");
    fis.addRule("IF Diversity IS High THEN Exploration IS Low AND Exploitation IS High");
    fis.addRule("IF Diversity IS Mid THEN Exploration IS Norm");

    auto fuzzyAdapter = std::make_shared<ga::fuzzy::FuzzyAlgorithmAdapter>(std::move(fis));

    ga::pso::PsoConfig psoFuzzyCfg;
    psoFuzzyCfg.search.dimension = 6;
    psoFuzzyCfg.search.iterations = 50;
    psoFuzzyCfg.search.seed = 88;
    psoFuzzyCfg.variant = ga::pso::PsoVariant::Constriction;
    psoFuzzyCfg.controller = fuzzyAdapter; // Adapts w and c1/c2 dynamically

    ga::pso::ParticleSwarmOptimizer fuzzyPso(psoFuzzyCfg);
    auto psoFuzzyRes = fuzzyPso.optimize(rastriginFitness);
    std::cout << "Fuzzy-Adapted PSO Rastrigin Best Fitness: " << psoFuzzyRes.bestFitness << "\n\n";

    // -------------------------------------------------------------------------
    // 4. FCM-Guided Swarm Topology Adaptation
    // -------------------------------------------------------------------------
    std::cout << "--- 4. FCM Swarm-Topology Analysis ---\n";
    ga::fuzzy::FcmSwarmAnalyzer fcmAnalyzer;
    std::vector<std::vector<double>> sampleSwarm = {
        {-2.0, -2.0}, {-1.9, -2.1}, {-2.1, -1.9},
        { 2.0,  2.0}, { 2.1,  1.9}, { 1.9,  2.1},
        { 0.0,  0.0}, { 0.1, -0.1}, {-0.1,  0.1}
    };
    auto fcmMetrics = fcmAnalyzer.analyze(sampleSwarm);
    std::cout << "FCM Partition Coefficient (V_PC): " << fcmMetrics.partitionCoefficient << "\n";
    std::cout << "FCM Partition Entropy (V_PE):     " << fcmMetrics.partitionEntropy << "\n";
    std::cout << "FCM Compactness Dispersion:       " << fcmMetrics.compactness << "\n\n";

    // -------------------------------------------------------------------------
    // 5. Interleaved Co-Evolutionary Hybrid (PSO <-> GSO Epochs)
    // -------------------------------------------------------------------------
    std::cout << "--- 5. Interleaved Co-Evolutionary Hybrid (PSO <-> GSO) ---\n";
    ga::pso::PsoConfig stage1Cfg;
    stage1Cfg.search.dimension = 6;
    stage1Cfg.search.iterations = 15;
    stage1Cfg.search.seed = 201;
    auto stage1 = std::make_shared<ga::pso::ParticleSwarmOptimizer>(stage1Cfg);

    ga::gso::GsoConfig stage2Cfg;
    stage2Cfg.search = stage1Cfg.search;
    stage2Cfg.search.seed = 202;
    auto stage2 = std::make_shared<ga::gso::GlowwormSwarmOptimizer>(stage2Cfg);

    ga::hybrid::InterleavedHybridConfig hybridCfg;
    hybridCfg.search = stage1Cfg.search;
    hybridCfg.epochs = 3;
    ga::hybrid::InterleavedHybridOptimizer interHybrid(stage1, stage2, hybridCfg);

    auto hybridRes = interHybrid.optimize(rastriginFitness);
    std::cout << "Interleaved Hybrid Rastrigin Best Fitness: " << hybridRes.bestFitness
              << " (Total Generations: " << hybridRes.generations << ")\n\n";

    // -------------------------------------------------------------------------
    // 6. Deeply Adaptive Fuzzy Crossover in Genetic Algorithm
    // -------------------------------------------------------------------------
    std::cout << "--- 6. Deeply Adaptive Fuzzy Crossover in GA ---\n";
    ga::Config gaCfg;
    gaCfg.dimension = 6;
    gaCfg.generations = 50;
    gaCfg.populationSize = 40;
    gaCfg.seed = 301;
    gaCfg.bounds = {-5.12, 5.12};

    ga::GeneticAlgorithm ga(gaCfg);
    ga.setCrossoverOperator(std::make_unique<ga::fuzzy::FuzzyAdaptiveCrossover>());

    auto gaRes = ga.run(sphereFitness);
    std::cout << "Fuzzy-Crossover GA Best Fitness: " << gaRes.bestFitness << "\n";

    std::cout << "\n========================================================\n";
    std::cout << "  Demo Completed Successfully!\n";
    std::cout << "========================================================\n";

    return 0;
}
