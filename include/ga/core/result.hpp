#pragma once

#include <vector>

namespace ga {
namespace core {

// Generic optimization result produced by any algorithm.
struct OptimizationResult {
    // Best solution found (genes / objective values).
    std::vector<double> bestSolution;

    // Fitness or first objective value of the best solution.
    double bestFitness = 0.0;

    // Per-generation best fitness history.
    std::vector<double> bestHistory;

    // Per-generation average fitness history.
    std::vector<double> avgHistory;

    // Pareto front (for multi-objective runs): each entry is a set of objective values.
    std::vector<std::vector<double>> paretoObjectives;

    // Corresponding genes for each Pareto-optimal solution.
    std::vector<std::vector<double>> paretoGenes;

    // Total number of fitness evaluations performed.
    std::size_t evaluations = 0;

    // Number of generations completed.
    std::size_t generations = 0;
};

} // namespace core
} // namespace ga
