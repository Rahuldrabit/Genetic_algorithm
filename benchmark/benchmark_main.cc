#include "ga_benchmark.h"
#include <iostream>
#include <string>

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --help, -h            Show this help message\n";
    std::cout << "  --iterations N        Number of benchmark iterations (default: 100)\n";
    std::cout << "  --csv                 Export results to CSV format\n";
    std::cout << "  --output FILE         Output file name (default: benchmark_results.txt)\n";
    std::cout << "  --operators           Run only operator benchmarks\n";
    std::cout << "  --functions           Run only function optimization benchmarks\n";
    std::cout << "  --scalability         Run only scalability benchmarks\n";
    std::cout << "  --all                 Run all benchmarks (default)\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << programName << " --all\n";
    std::cout << "  " << programName << " --operators --csv\n";
    std::cout << "  " << programName << " --functions --iterations 50\n";
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config;
    config.benchmarkIterations = 100;
    config.warmupIterations = 5;
    config.verbose = true;
    config.csvOutput = false;
    config.outputFile = "benchmark_results.txt";

    bool runOperators = false;
    bool runFunctions = false;
    bool runScalability = false;
    bool runAll = true;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.benchmarkIterations = std::stoul(argv[++i]);
        } else if (arg == "--csv") {
            config.csvOutput = true;
        } else if (arg == "--output" && i + 1 < argc) {
            config.outputFile = argv[++i];
        } else if (arg == "--operators") {
            runOperators = true;
            runAll = false;
        } else if (arg == "--functions") {
            runFunctions = true;
            runAll = false;
        } else if (arg == "--scalability") {
            runScalability = true;
            runAll = false;
        } else if (arg == "--all") {
            runAll = true;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::cerr << "Use --help for usage information.\n";
            return 1;
        }
    }

    try {
        GABenchmark benchmark(config);

        if (runAll) {
            benchmark.runAllBenchmarks();
        } else {
            if (runOperators) {
                benchmark.runOperatorBenchmarks();
            }
            if (runFunctions) {
                benchmark.runFunctionBenchmarks();
            }
            if (runScalability) {
                benchmark.runScalabilityBenchmarks();
            }
            benchmark.generateReport();
            if (config.csvOutput) {
                benchmark.exportToCSV("benchmark_results.csv");
            }
        }

        std::cout << "\n✓ Benchmarking completed successfully!\n";
        std::cout << "Results saved to: " << config.outputFile << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error during benchmarking: " << e.what() << "\n";
        return 1;
    }
}
