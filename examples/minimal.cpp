#include <iostream>
#include <vector>
#include <cmath>
#include "ga/genetic_algorithm.hpp"

namespace {
constexpr double kPi = 3.14159265358979323846;
}

static double rastrigin(const std::vector<double>& x) {
    const double A = 10.0;
    double sum = A * static_cast<double>(x.size());
    for (double xi : x) sum += xi*xi - A*std::cos(2*kPi*xi);
    // Convert minimization objective to maximization fitness
    double f = sum;
    return 1000.0 / (1.0 + f);
}

int main() {
    ga::Config cfg;
    cfg.populationSize = 60;
    cfg.generations = 100;
    cfg.dimension = 10;
    cfg.crossoverRate = 0.9;
    cfg.mutationRate = 0.1;
    cfg.bounds = {-5.12, 5.12};

    ga::GeneticAlgorithm gae(cfg);
    auto res = gae.run(rastrigin);

    std::cout << "Best fitness: " << res.bestFitness << "\nBest genes: ";
    for (double g : res.bestGenes) std::cout << g << ' ';
    std::cout << '\n';
    return 0;
}
