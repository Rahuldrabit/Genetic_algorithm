#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <memory>

// Benchmark results structure
struct BenchmarkResult {
    std::string name;
    std::string category;
    double avgExecutionTime;      // microseconds
    double minExecutionTime;
    double maxExecutionTime;
    size_t iterations;
    double throughput;             // operations per second
    double standardDeviation;
    bool success;
    std::string errorMessage;
};

// Operator benchmark results
struct OperatorBenchmark {
    std::string operatorName;
    std::string operatorType;      // "crossover", "mutation", "selection"
    double avgTime;                 // microseconds per operation
    size_t operationsPerSecond;
    size_t iterations;
    std::string representation;     // "binary", "real", "integer", "permutation"
};

// GA function benchmark results
struct FunctionBenchmark {
    std::string functionName;
    double bestFitness;
    double avgFitness;
    size_t generationsToConverge;
    double totalExecutionTime;     // milliseconds
    std::vector<double> bestSolution;
    std::vector<double> convergenceHistory;
};

// Benchmark configuration
struct BenchmarkConfig {
    size_t warmupIterations = 5;
    size_t benchmarkIterations = 100;
    bool verbose = true;
    bool csvOutput = false;
    std::string outputFile = "benchmark_results.txt";
};

// Main benchmark class
class GABenchmark {
public:
    explicit GABenchmark(const BenchmarkConfig& config);

    // Run all benchmarks
    void runAllBenchmarks();

    // Run specific benchmark categories
    void runOperatorBenchmarks();
    void runFunctionBenchmarks();
    void runScalabilityBenchmarks();

    // Generate reports
    void generateReport();
    void exportToCSV(const std::string& filename);

private:
    BenchmarkConfig config_;
    std::vector<OperatorBenchmark> operatorResults_;
    std::vector<FunctionBenchmark> functionResults_;
    std::vector<BenchmarkResult> scalabilityResults_;

    // Crossover operator benchmarks
    void benchmarkCrossoverOperators();
    void benchmarkBinaryCrossover();
    void benchmarkRealCrossover();
    void benchmarkIntegerCrossover();
    void benchmarkPermutationCrossover();

    // Mutation operator benchmarks
    void benchmarkMutationOperators();
    void benchmarkBinaryMutation();
    void benchmarkRealMutation();
    void benchmarkIntegerMutation();
    void benchmarkPermutationMutation();

    // Selection operator benchmarks
    void benchmarkSelectionOperators();

    // Function optimization benchmarks
    void benchmarkOptimizationFunctions();
    void benchmarkRastrigin();
    void benchmarkAckley();
    void benchmarkSchwefel();
    void benchmarkRosenbrock();
    void benchmarkSphere();

    // Scalability tests
    void benchmarkPopulationScaling();
    void benchmarkDimensionScaling();
    void benchmarkGenerationScaling();

    // Utility functions
    template<typename Func>
    BenchmarkResult timeBenchmark(const std::string& name, const std::string& category, Func&& func);

    void printHeader(const std::string& title);
    void printResult(const BenchmarkResult& result);
    void printOperatorResult(const OperatorBenchmark& result);
    void printFunctionResult(const FunctionBenchmark& result);
};
