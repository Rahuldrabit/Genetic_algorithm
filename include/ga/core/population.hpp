#pragma once

#include <vector>
#include "ga/core/individual.hpp"

namespace ga {
namespace core {

// A typed population is just a collection of Individuals.
// This header provides a named alias and utilities used by algorithm implementations.
using Population = std::vector<ga::Individual>;

// Returns the index of the Individual with the best (lowest) first objective.
// If the population is empty, returns pop.size() as an out-of-range sentinel.
inline std::size_t bestIndex(const Population& pop) {
    if (pop.empty()) {
        return pop.size();
    }
    std::size_t best = 0;
    for (std::size_t i = 1; i < pop.size(); ++i) {
        const auto& a = pop[i].evaluation.objectives;
        const auto& b = pop[best].evaluation.objectives;
        if (!a.empty() && !b.empty() && a[0] < b[0]) {
            best = i;
        }
    }
    return best;
}

// Returns a sub-population of the top-k individuals sorted by first objective (ascending).
inline Population topK(const Population& pop, std::size_t k) {
    Population sorted = pop;
    if (k > sorted.size()) {
        k = sorted.size();
    }
    std::partial_sort(
        sorted.begin(),
        sorted.begin() + static_cast<std::ptrdiff_t>(k),
        sorted.end(),
        [](const ga::Individual& a, const ga::Individual& b) {
            if (a.evaluation.objectives.empty()) return false;
            if (b.evaluation.objectives.empty()) return true;
            return a.evaluation.objectives[0] < b.evaluation.objectives[0];
        });
    sorted.resize(k);
    return sorted;
}

} // namespace core
} // namespace ga
