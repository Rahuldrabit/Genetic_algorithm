#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ga/core/individual.hpp"

namespace ga {
namespace moea {

class Spea2 {
public:
    std::vector<double> strengthFitness(const std::vector<ga::Individual>& population) const {
        if (population.empty()) {
            return {};
        }
        const std::size_t n = population.size();
        std::vector<double> fitness(n, 0.0);
        std::vector<double> strength(n, 0.0);

        auto dominates = [](const ga::Individual& a, const ga::Individual& b) {
            if (a.evaluation.objectives.size() != b.evaluation.objectives.size() ||
                a.evaluation.objectives.empty()) {
                return false;
            }
            bool better = false;
            for (std::size_t i = 0; i < a.evaluation.objectives.size(); ++i) {
                if (a.evaluation.objectives[i] > b.evaluation.objectives[i]) {
                    return false;
                }
                better |= a.evaluation.objectives[i] < b.evaluation.objectives[i];
            }
            return better;
        };

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                if (i != j && dominates(population[i], population[j])) {
                    strength[i] += 1.0;
                }
            }
        }

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                if (i != j && dominates(population[j], population[i])) {
                    fitness[i] += strength[j];
                }
            }
        }

        // Density estimate via nearest-neighbor distance in objective space.
        for (std::size_t i = 0; i < n; ++i) {
            double nearest = std::numeric_limits<double>::infinity();
            for (std::size_t j = 0; j < n; ++j) {
                if (i == j) {
                    continue;
                }
                double dist2 = 0.0;
                for (std::size_t k = 0; k < population[i].evaluation.objectives.size(); ++k) {
                    const double d = population[i].evaluation.objectives[k] - population[j].evaluation.objectives[k];
                    dist2 += d * d;
                }
                nearest = std::min(nearest, std::sqrt(dist2));
            }
            if (std::isfinite(nearest)) {
                fitness[i] += 1.0 / (nearest + 2.0);
            }
        }

        return fitness;
    }

    std::vector<ga::Individual>
    environmentalSelect(const std::vector<ga::Individual>& combined,
                        std::size_t targetSize) const {
        if (targetSize == 0 || combined.empty()) {
            return {};
        }
        auto fit = strengthFitness(combined);
        std::vector<std::size_t> order(combined.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }

        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return fit[a] < fit[b]; // lower is better in SPEA2
        });

        std::vector<ga::Individual> out;
        out.reserve(targetSize);
        for (std::size_t i = 0; i < targetSize && i < order.size(); ++i) {
            out.push_back(combined[order[i]]);
        }
        return out;
    }
};

} // namespace moea
} // namespace ga
