#pragma once

#include <algorithm>
#include <functional>
#include <vector>

#include "ga/config.hpp"
#include "ga/genetic_algorithm.hpp"

namespace ga {
namespace hybrid {

class HybridOptimizer {
public:
    using LocalSearch = std::function<void(std::vector<double>&)>;

    explicit HybridOptimizer(ga::Config config)
        : config_(std::move(config)) {}

    ga::Result run(const ga::Fitness& fitness,
                   const LocalSearch& localSearch,
                   std::size_t localSearchRestarts = 5) const {
        ga::GeneticAlgorithm ga(config_);
        ga::Result res = ga.run(fitness);

        std::vector<double> best = res.bestGenes;
        double bestFitness = res.bestFitness;

        for (std::size_t i = 0; i < localSearchRestarts; ++i) {
            std::vector<double> candidate = best;
            if (localSearch) {
                localSearch(candidate);
            }
            const double f = fitness(candidate);
            if (f > bestFitness) {
                bestFitness = f;
                best = std::move(candidate);
            }
        }

        res.bestGenes = std::move(best);
        res.bestFitness = bestFitness;
        if (!res.bestHistory.empty()) {
            res.bestHistory.back() = bestFitness;
        }
        return res;
    }

private:
    ga::Config config_;
};

} // namespace hybrid
} // namespace ga
