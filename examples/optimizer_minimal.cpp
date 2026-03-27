#include <iostream>
#include <vector>

#include "ga/api/optimizer.hpp"

int main() {
    ga::Config cfg;
    cfg.populationSize = 40;
    cfg.generations = 40;
    cfg.dimension = 6;
    cfg.bounds = {-3.0, 3.0};
    cfg.seed = 77;

    ga::api::Optimizer optimizer;
    optimizer.withConfig(cfg);

    auto result = optimizer.optimize([](const std::vector<double>& x) {
        double sum = 0.0;
        for (double v : x) {
            sum += v * v;
        }
        return 1000.0 / (1.0 + sum);
    });

    std::cout << "Best fitness: " << result.bestFitness << "\n";

    auto mo = optimizer.optimizeMultiObjective({
        [](const std::vector<double>& x) {
            double s = 0.0;
            for (double v : x) s += v * v;
            return s;
        },
        [](const std::vector<double>& x) {
            double s = 0.0;
            for (double v : x) s += (v - 1.0) * (v - 1.0);
            return s;
        }
    }, 40, 30);

    std::cout << "Pareto points: " << mo.paretoGenes.size() << "\n";

    auto mo3 = optimizer.optimizeMultiObjectiveNsga3({
        [](const std::vector<double>& x) {
            double s = 0.0;
            for (double v : x) s += v * v;
            return s;
        },
        [](const std::vector<double>& x) {
            double s = 0.0;
            for (double v : x) s += (v - 1.0) * (v - 1.0);
            return s;
        }
    },
                                                     40,
                                                     30,
                                                     8);
    std::cout << "NSGA-III Pareto points: " << mo3.paretoGenes.size() << "\n";
    return 0;
}
