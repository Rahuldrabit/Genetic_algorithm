#pragma once

#include <vector>
#include <string>
#include <functional>

namespace ga {

using Fitness = std::function<double(const std::vector<double>&)>; // higher is better

struct Bounds {
    double lower = -1.0;
    double upper = 1.0;
};

struct Config {
    int populationSize = 50;
    int generations = 100;
    int dimension = 10;

    double crossoverRate = 0.8;
    double mutationRate = 0.1;

    Bounds bounds{ -5.12, 5.12 };

    // Elitism: fraction [0,1]
    double eliteRatio = 0.05;

    // Random seed (0 uses random_device)
    unsigned seed = 0;
};

struct Result {
    std::vector<double> bestGenes;
    double bestFitness = -1e300;
    std::vector<double> bestHistory; // best per generation
    std::vector<double> avgHistory;  // average per generation
};

} // namespace ga
