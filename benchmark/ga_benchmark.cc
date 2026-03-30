#include "ga_benchmark.h"
#include <ga/genetic_algorithm.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <sstream>

// MSVC does not guarantee `M_PI` is defined unless special macros are set.
// Define it here to keep the benchmark code portable.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Include all operators for testing
#include "crossover/one_point_crossover.h"
#include "crossover/two_point_crossover.h"
#include "crossover/uniform_crossover.h"
#include "crossover/blend_crossover.h"
#include "crossover/simulated_binary_crossover.h"
#include "crossover/order_crossover.h"
#include "crossover/partially_mapped_crossover.h"
#include "crossover/cycle_crossover.h"
#include "mutation/bit_flip_mutation.h"
#include "mutation/gaussian_mutation.h"
#include "mutation/uniform_mutation.h"
#include "mutation/swap_mutation.h"
#include "mutation/inversion_mutation.h"
#include "mutation/random_resetting_mutation.h"
#include "mutation/creep_mutation.h"
#include "selection-operator/tournament_selection.h"
#include "selection-operator/roulette_wheel_selection.h"
#include "selection-operator/rank_selection.h"

// Benchmark fitness functions
namespace BenchmarkFunctions {

// Sphere function: f(x) = sum(xi^2)
// Global minimum: f(0, ..., 0) = 0
// Domain: [-5.12, 5.12]^n
double sphere(const std::vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += xi * xi;
    }
    return 1000.0 / (1.0 + sum);
}

// Rastrigin function
// Global minimum: f(0, ..., 0) = 0
// Domain: [-5.12, 5.12]^n
double rastrigin(const std::vector<double>& x) {
    const double A = 10.0;
    double sum = A * x.size();
    for (double xi : x) {
        sum += xi * xi - A * std::cos(2.0 * M_PI * xi);
    }
    return 1000.0 / (1.0 + sum);
}

// Ackley function
// Global minimum: f(0, ..., 0) = 0
// Domain: [-32.768, 32.768]^n
double ackley(const std::vector<double>& x) {
    const double a = 20.0;
    const double b = 0.2;
    const double c = 2.0 * M_PI;
    size_t n = x.size();

    double sum1 = 0.0, sum2 = 0.0;
    for (double xi : x) {
        sum1 += xi * xi;
        sum2 += std::cos(c * xi);
    }

    double term1 = -a * std::exp(-b * std::sqrt(sum1 / n));
    double term2 = -std::exp(sum2 / n);
    double result = term1 + term2 + a + std::exp(1.0);

    return 1000.0 / (1.0 + result);
}

// Schwefel function
// Global minimum: f(420.9687, ..., 420.9687) = 0
// Domain: [-500, 500]^n
double schwefel(const std::vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += xi * std::sin(std::sqrt(std::abs(xi)));
    }
    double result = 418.9829 * x.size() - sum;
    return 1000.0 / (1.0 + result);
}

// Rosenbrock function
// Global minimum: f(1, ..., 1) = 0
// Domain: [-5, 10]^n
double rosenbrock(const std::vector<double>& x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size() - 1; ++i) {
        double term1 = x[i + 1] - x[i] * x[i];
        double term2 = 1.0 - x[i];
        sum += 100.0 * term1 * term1 + term2 * term2;
    }
    return 1000.0 / (1.0 + sum);
}

} // namespace BenchmarkFunctions

GABenchmark::GABenchmark(const BenchmarkConfig& config) : config_(config) {}

void GABenchmark::runAllBenchmarks() {
    std::cout << "\n";
    std::cout << "========================================================\n";
    std::cout << "  GENETIC ALGORITHM COMPREHENSIVE BENCHMARK SUITE\n";
    std::cout << "========================================================\n";
    std::cout << "\n";

    runOperatorBenchmarks();
    runFunctionBenchmarks();
    runScalabilityBenchmarks();

    generateReport();

    if (config_.csvOutput) {
        exportToCSV("benchmark_results.csv");
    }
}

void GABenchmark::runOperatorBenchmarks() {
    printHeader("OPERATOR BENCHMARKS");
    benchmarkCrossoverOperators();
    benchmarkMutationOperators();
    benchmarkSelectionOperators();
}

void GABenchmark::benchmarkCrossoverOperators() {
    std::cout << "\n--- Crossover Operators ---\n\n";
    benchmarkBinaryCrossover();
    benchmarkRealCrossover();
    benchmarkIntegerCrossover();
    benchmarkPermutationCrossover();
}

void GABenchmark::benchmarkBinaryCrossover() {
    const size_t chromSize = 100;
    BitString parent1(chromSize, false);
    BitString parent2(chromSize, true);

    // Fill with alternating pattern
    for (size_t i = 0; i < chromSize; ++i) {
        parent1[i] = (i % 2 == 0);
        parent2[i] = (i % 2 == 1);
    }

    // One-Point Crossover
    {
        OnePointCrossover op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            auto result = op.crossover(parent1, parent2);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "OnePointCrossover";
        result.operatorType = "crossover";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "binary";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }

    // Two-Point Crossover
    {
        TwoPointCrossover op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            auto result = op.crossover(parent1, parent2);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "TwoPointCrossover";
        result.operatorType = "crossover";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "binary";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }

    // Uniform Crossover
    {
        UniformCrossover op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            auto result = op.crossover(parent1, parent2);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "UniformCrossover";
        result.operatorType = "crossover";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "binary";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }
}

void GABenchmark::benchmarkRealCrossover() {
    const size_t chromSize = 10;
    RealVector parent1(chromSize, 1.0);
    RealVector parent2(chromSize, 5.0);

    // Blend Crossover (BLX-α)
    {
        BlendCrossover op(0.5);
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            auto result = op.crossover(parent1, parent2);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "BlendCrossover (BLX-α)";
        result.operatorType = "crossover";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "real";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }

    // Simulated Binary Crossover (SBX)
    {
        SimulatedBinaryCrossover op(2.0);
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            auto result = op.crossover(parent1, parent2);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "SimulatedBinaryCrossover (SBX)";
        result.operatorType = "crossover";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "real";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }
}

void GABenchmark::benchmarkIntegerCrossover() {
    const size_t chromSize = 20;
    IntVector parent1(chromSize, 10);
    IntVector parent2(chromSize, 50);

    // One-Point Crossover for integers
    {
        OnePointCrossover op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            auto result = op.crossover(parent1, parent2);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "OnePointCrossover";
        result.operatorType = "crossover";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "integer";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }
}

void GABenchmark::benchmarkPermutationCrossover() {
    const size_t chromSize = 20;
    Permutation parent1(chromSize);
    Permutation parent2(chromSize);

    for (size_t i = 0; i < chromSize; ++i) {
        parent1[i] = static_cast<int>(i);
        parent2[i] = static_cast<int>(chromSize - i - 1);
    }

    // Order Crossover (OX)
    {
        OrderCrossover op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            auto result = op.crossover(parent1, parent2);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "OrderCrossover (OX)";
        result.operatorType = "crossover";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "permutation";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }

    // Partially Mapped Crossover (PMX)
    {
        PartiallyMappedCrossover op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            auto result = op.crossover(parent1, parent2);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "PartiallyMappedCrossover (PMX)";
        result.operatorType = "crossover";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "permutation";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }

    // Cycle Crossover (CX)
    {
        CycleCrossover op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            auto result = op.crossover(parent1, parent2);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "CycleCrossover (CX)";
        result.operatorType = "crossover";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "permutation";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }
}

void GABenchmark::benchmarkMutationOperators() {
    std::cout << "\n--- Mutation Operators ---\n\n";
    benchmarkBinaryMutation();
    benchmarkRealMutation();
    benchmarkIntegerMutation();
    benchmarkPermutationMutation();
}

void GABenchmark::benchmarkBinaryMutation() {
    const size_t chromSize = 100;
    BitString chromosome(chromSize, false);
    const double mutationRate = 0.1;

    // Bit Flip Mutation
    {
        BitFlipMutation op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            BitString temp = chromosome;
            op.mutate(temp, mutationRate);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "BitFlipMutation";
        result.operatorType = "mutation";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "binary";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }
}

void GABenchmark::benchmarkRealMutation() {
    const size_t chromSize = 10;
    RealVector chromosome(chromSize, 0.0);
    RealVector lowerBounds(chromSize, -5.0);
    RealVector upperBounds(chromSize, 5.0);
    const double mutationRate = 0.1;

    // Gaussian Mutation
    {
        GaussianMutation op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            RealVector temp = chromosome;
            op.mutate(temp, mutationRate, 1.0, lowerBounds, upperBounds);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "GaussianMutation";
        result.operatorType = "mutation";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "real";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }

    // Uniform Mutation
    {
        UniformMutation op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            RealVector temp = chromosome;
            op.mutate(temp, mutationRate, lowerBounds, upperBounds);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "UniformMutation";
        result.operatorType = "mutation";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "real";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }
}

void GABenchmark::benchmarkIntegerMutation() {
    const size_t chromSize = 20;
    IntVector chromosome(chromSize, 25);
    IntVector lowerBounds(chromSize, 0);
    IntVector upperBounds(chromSize, 100);
    const double mutationRate = 0.1;

    // Random Resetting Mutation
    {
        RandomResettingMutation op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            IntVector temp = chromosome;
            op.mutate(temp, mutationRate, 0, 100);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "RandomResettingMutation";
        result.operatorType = "mutation";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "integer";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }

    // Creep Mutation
    {
        CreepMutation op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            IntVector temp = chromosome;
            op.mutate(temp, mutationRate, 5, 0, 100);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "CreepMutation";
        result.operatorType = "mutation";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "integer";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }
}

void GABenchmark::benchmarkPermutationMutation() {
    const size_t chromSize = 20;
    Permutation chromosome(chromSize);
    const double mutationRate = 0.1;
    for (size_t i = 0; i < chromSize; ++i) {
        chromosome[i] = static_cast<int>(i);
    }

    // Swap Mutation
    {
        SwapMutation op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            Permutation temp = chromosome;
            op.mutate(temp, mutationRate);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "SwapMutation";
        result.operatorType = "mutation";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "permutation";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }

    // Inversion Mutation
    {
        InversionMutation op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            Permutation temp = chromosome;
            op.mutate(temp, mutationRate);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "InversionMutation";
        result.operatorType = "mutation";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "permutation";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }
}

void GABenchmark::benchmarkSelectionOperators() {
    std::cout << "\n--- Selection Operators ---\n\n";

    const size_t popSize = 100;
    std::vector<Individual> population(popSize);
    for (size_t i = 0; i < popSize; ++i) {
        population[i].fitness = static_cast<double>(i + 1);
    }

    // Tournament Selection
    {
        TournamentSelection op(3);
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            auto selected = op.select(population, popSize);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "TournamentSelection (k=3)";
        result.operatorType = "selection";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "any";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }

    // Roulette Wheel Selection
    {
        RouletteWheelSelection op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            auto selected = op.select(population, popSize);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "RouletteWheelSelection";
        result.operatorType = "selection";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "any";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }

    // Rank Selection
    {
        RankSelection op;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < config_.benchmarkIterations; ++i) {
            auto selected = op.select(population, popSize);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        OperatorBenchmark result;
        result.operatorName = "RankSelection";
        result.operatorType = "selection";
        result.avgTime = static_cast<double>(duration) / config_.benchmarkIterations;
        result.operationsPerSecond = static_cast<size_t>(1e6 / result.avgTime);
        result.iterations = config_.benchmarkIterations;
        result.representation = "any";
        operatorResults_.push_back(result);
        printOperatorResult(result);
    }
}

void GABenchmark::runFunctionBenchmarks() {
    printHeader("OPTIMIZATION FUNCTION BENCHMARKS");
    benchmarkOptimizationFunctions();
}

void GABenchmark::benchmarkOptimizationFunctions() {
    benchmarkSphere();
    benchmarkRastrigin();
    benchmarkAckley();
    benchmarkSchwefel();
    benchmarkRosenbrock();
}

void GABenchmark::benchmarkSphere() {
    ga::Config cfg;
    cfg.populationSize = 50;
    cfg.generations = 100;
    cfg.dimension = 10;
    cfg.bounds = {-5.12, 5.12};
    cfg.crossoverRate = 0.8;
    cfg.mutationRate = 0.1;
    cfg.eliteRatio = 0.1;

    ga::GeneticAlgorithm alg(cfg);

    auto start = std::chrono::high_resolution_clock::now();
    ga::Result result = alg.run(BenchmarkFunctions::sphere);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    FunctionBenchmark bench;
    bench.functionName = "Sphere";
    bench.bestFitness = result.bestFitness;
    bench.avgFitness = std::accumulate(result.avgHistory.begin(), result.avgHistory.end(), 0.0) / result.avgHistory.size();
    bench.generationsToConverge = result.bestHistory.size();
    bench.totalExecutionTime = static_cast<double>(duration);
    bench.bestSolution = result.bestGenes;
    bench.convergenceHistory = result.bestHistory;
    functionResults_.push_back(bench);
    printFunctionResult(bench);
}

void GABenchmark::benchmarkRastrigin() {
    ga::Config cfg;
    cfg.populationSize = 60;
    cfg.generations = 200;
    cfg.dimension = 10;
    cfg.bounds = {-5.12, 5.12};
    cfg.crossoverRate = 0.8;
    cfg.mutationRate = 0.1;
    cfg.eliteRatio = 0.1;

    ga::GeneticAlgorithm alg(cfg);

    auto start = std::chrono::high_resolution_clock::now();
    ga::Result result = alg.run(BenchmarkFunctions::rastrigin);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    FunctionBenchmark bench;
    bench.functionName = "Rastrigin";
    bench.bestFitness = result.bestFitness;
    bench.avgFitness = std::accumulate(result.avgHistory.begin(), result.avgHistory.end(), 0.0) / result.avgHistory.size();
    bench.generationsToConverge = result.bestHistory.size();
    bench.totalExecutionTime = static_cast<double>(duration);
    bench.bestSolution = result.bestGenes;
    bench.convergenceHistory = result.bestHistory;
    functionResults_.push_back(bench);
    printFunctionResult(bench);
}

void GABenchmark::benchmarkAckley() {
    ga::Config cfg;
    cfg.populationSize = 60;
    cfg.generations = 150;
    cfg.dimension = 10;
    cfg.bounds = {-32.768, 32.768};
    cfg.crossoverRate = 0.8;
    cfg.mutationRate = 0.1;
    cfg.eliteRatio = 0.1;

    ga::GeneticAlgorithm alg(cfg);

    auto start = std::chrono::high_resolution_clock::now();
    ga::Result result = alg.run(BenchmarkFunctions::ackley);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    FunctionBenchmark bench;
    bench.functionName = "Ackley";
    bench.bestFitness = result.bestFitness;
    bench.avgFitness = std::accumulate(result.avgHistory.begin(), result.avgHistory.end(), 0.0) / result.avgHistory.size();
    bench.generationsToConverge = result.bestHistory.size();
    bench.totalExecutionTime = static_cast<double>(duration);
    bench.bestSolution = result.bestGenes;
    bench.convergenceHistory = result.bestHistory;
    functionResults_.push_back(bench);
    printFunctionResult(bench);
}

void GABenchmark::benchmarkSchwefel() {
    ga::Config cfg;
    cfg.populationSize = 80;
    cfg.generations = 200;
    cfg.dimension = 10;
    cfg.bounds = {-500.0, 500.0};
    cfg.crossoverRate = 0.8;
    cfg.mutationRate = 0.1;
    cfg.eliteRatio = 0.1;

    ga::GeneticAlgorithm alg(cfg);

    auto start = std::chrono::high_resolution_clock::now();
    ga::Result result = alg.run(BenchmarkFunctions::schwefel);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    FunctionBenchmark bench;
    bench.functionName = "Schwefel";
    bench.bestFitness = result.bestFitness;
    bench.avgFitness = std::accumulate(result.avgHistory.begin(), result.avgHistory.end(), 0.0) / result.avgHistory.size();
    bench.generationsToConverge = result.bestHistory.size();
    bench.totalExecutionTime = static_cast<double>(duration);
    bench.bestSolution = result.bestGenes;
    bench.convergenceHistory = result.bestHistory;
    functionResults_.push_back(bench);
    printFunctionResult(bench);
}

void GABenchmark::benchmarkRosenbrock() {
    ga::Config cfg;
    cfg.populationSize = 100;
    cfg.generations = 300;
    cfg.dimension = 10;
    cfg.bounds = {-5.0, 10.0};
    cfg.crossoverRate = 0.8;
    cfg.mutationRate = 0.1;
    cfg.eliteRatio = 0.1;

    ga::GeneticAlgorithm alg(cfg);

    auto start = std::chrono::high_resolution_clock::now();
    ga::Result result = alg.run(BenchmarkFunctions::rosenbrock);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    FunctionBenchmark bench;
    bench.functionName = "Rosenbrock";
    bench.bestFitness = result.bestFitness;
    bench.avgFitness = std::accumulate(result.avgHistory.begin(), result.avgHistory.end(), 0.0) / result.avgHistory.size();
    bench.generationsToConverge = result.bestHistory.size();
    bench.totalExecutionTime = static_cast<double>(duration);
    bench.bestSolution = result.bestGenes;
    bench.convergenceHistory = result.bestHistory;
    functionResults_.push_back(bench);
    printFunctionResult(bench);
}

void GABenchmark::runScalabilityBenchmarks() {
    printHeader("SCALABILITY BENCHMARKS");
    benchmarkPopulationScaling();
    benchmarkDimensionScaling();
}

void GABenchmark::benchmarkPopulationScaling() {
    std::cout << "\n--- Population Size Scaling ---\n\n";

    std::vector<int> popSizes = {10, 25, 50, 100, 200};

    for (int popSize : popSizes) {
        ga::Config cfg;
        cfg.populationSize = popSize;
        cfg.generations = 50;
        cfg.dimension = 10;
        cfg.bounds = {-5.12, 5.12};

        ga::GeneticAlgorithm alg(cfg);

        auto start = std::chrono::high_resolution_clock::now();
        ga::Result result = alg.run(BenchmarkFunctions::sphere);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        BenchmarkResult bench;
        bench.name = "PopSize=" + std::to_string(popSize);
        bench.category = "scalability";
        bench.avgExecutionTime = static_cast<double>(duration);
        bench.success = true;

        std::cout << std::setw(15) << bench.name << " | "
                  << std::setw(10) << std::fixed << std::setprecision(2)
                  << bench.avgExecutionTime << " ms | "
                  << "Best Fitness: " << std::setprecision(6) << result.bestFitness << "\n";

        scalabilityResults_.push_back(bench);
    }
}

void GABenchmark::benchmarkDimensionScaling() {
    std::cout << "\n--- Problem Dimension Scaling ---\n\n";

    std::vector<int> dimensions = {5, 10, 20, 30, 50};

    for (int dim : dimensions) {
        ga::Config cfg;
        cfg.populationSize = 50;
        cfg.generations = 50;
        cfg.dimension = dim;
        cfg.bounds = {-5.12, 5.12};

        ga::GeneticAlgorithm alg(cfg);

        auto start = std::chrono::high_resolution_clock::now();
        ga::Result result = alg.run(BenchmarkFunctions::sphere);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        BenchmarkResult bench;
        bench.name = "Dimension=" + std::to_string(dim);
        bench.category = "scalability";
        bench.avgExecutionTime = static_cast<double>(duration);
        bench.success = true;

        std::cout << std::setw(15) << bench.name << " | "
                  << std::setw(10) << std::fixed << std::setprecision(2)
                  << bench.avgExecutionTime << " ms | "
                  << "Best Fitness: " << std::setprecision(6) << result.bestFitness << "\n";

        scalabilityResults_.push_back(bench);
    }
}

void GABenchmark::generateReport() {
    std::ofstream outFile(config_.outputFile);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open output file: " << config_.outputFile << "\n";
        return;
    }

    outFile << "========================================================\n";
    outFile << "  GENETIC ALGORITHM BENCHMARK RESULTS\n";
    outFile << "========================================================\n\n";

    outFile << "Configuration:\n";
    outFile << "  Warmup Iterations: " << config_.warmupIterations << "\n";
    outFile << "  Benchmark Iterations: " << config_.benchmarkIterations << "\n\n";

    outFile << "--- Operator Performance Summary ---\n\n";
    for (const auto& result : operatorResults_) {
        outFile << result.operatorType << " | " << result.operatorName
                << " [" << result.representation << "]\n";
        outFile << "  Avg Time: " << std::fixed << std::setprecision(3)
                << result.avgTime << " μs\n";
        outFile << "  Throughput: " << result.operationsPerSecond << " ops/sec\n\n";
    }

    outFile << "\n--- Function Optimization Summary ---\n\n";
    for (const auto& result : functionResults_) {
        outFile << result.functionName << " Function:\n";
        outFile << "  Best Fitness: " << std::fixed << std::setprecision(6)
                << result.bestFitness << "\n";
        outFile << "  Avg Fitness: " << result.avgFitness << "\n";
        outFile << "  Generations: " << result.generationsToConverge << "\n";
        outFile << "  Execution Time: " << std::setprecision(2)
                << result.totalExecutionTime << " ms\n\n";
    }

    outFile << "\n--- Scalability Results ---\n\n";
    for (const auto& result : scalabilityResults_) {
        outFile << result.name << ": " << std::fixed << std::setprecision(2)
                << result.avgExecutionTime << " ms\n";
    }

    outFile.close();
    std::cout << "\n[✓] Benchmark report saved to: " << config_.outputFile << "\n";
}

void GABenchmark::exportToCSV(const std::string& filename) {
    std::ofstream csvFile(filename);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open CSV file: " << filename << "\n";
        return;
    }

    // Operator results CSV
    csvFile << "Type,Operator,Representation,AvgTime(μs),Throughput(ops/s)\n";
    for (const auto& result : operatorResults_) {
        csvFile << result.operatorType << ","
                << result.operatorName << ","
                << result.representation << ","
                << result.avgTime << ","
                << result.operationsPerSecond << "\n";
    }

    csvFile << "\n";

    // Function results CSV
    csvFile << "Function,BestFitness,AvgFitness,Generations,Time(ms)\n";
    for (const auto& result : functionResults_) {
        csvFile << result.functionName << ","
                << result.bestFitness << ","
                << result.avgFitness << ","
                << result.generationsToConverge << ","
                << result.totalExecutionTime << "\n";
    }

    csvFile.close();
    std::cout << "[✓] CSV results saved to: " << filename << "\n";
}

void GABenchmark::printHeader(const std::string& title) {
    std::cout << "\n";
    std::cout << "========================================================\n";
    std::cout << "  " << title << "\n";
    std::cout << "========================================================\n";
}

void GABenchmark::printOperatorResult(const OperatorBenchmark& result) {
    std::cout << std::setw(35) << std::left << result.operatorName
              << " [" << std::setw(12) << result.representation << "] | "
              << std::setw(8) << std::right << std::fixed << std::setprecision(2)
              << result.avgTime << " μs | "
              << std::setw(10) << result.operationsPerSecond << " ops/s\n";
}

void GABenchmark::printFunctionResult(const FunctionBenchmark& result) {
    std::cout << "\n" << result.functionName << " Function:\n";
    std::cout << "  Best Fitness: " << std::fixed << std::setprecision(6)
              << result.bestFitness << "\n";
    std::cout << "  Average Fitness: " << result.avgFitness << "\n";
    std::cout << "  Generations: " << result.generationsToConverge << "\n";
    std::cout << "  Execution Time: " << std::setprecision(2)
              << result.totalExecutionTime << " ms\n";
}
