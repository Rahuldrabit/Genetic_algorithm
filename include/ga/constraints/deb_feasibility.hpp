#pragma once

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace ga {
namespace constraints {

struct DebProfile {
    double fitness = 0.0;
    double violation = 0.0; // 0.0 means feasible, > 0.0 means infeasible
    bool isFeasible = true;
};

// Returns true if candidate 'a' dominates or is preferred over candidate 'b' according to Deb's rules (Deb 2000).
inline bool debIsBetter(const DebProfile& a, const DebProfile& b) {
    // Rule 1: Any feasible solution is preferred to any infeasible solution
    if (a.isFeasible && !b.isFeasible) {
        return true;
    }
    if (!a.isFeasible && b.isFeasible) {
        return false;
    }

    // Rule 2: Between two feasible solutions, the one with better fitness is preferred
    if (a.isFeasible && b.isFeasible) {
        return a.fitness > b.fitness;
    }

    // Rule 3: Between two infeasible solutions, the one with smaller constraint violation is preferred
    return a.violation < b.violation;
}

// Deb's tournament selector
template <typename T>
inline std::size_t debTournamentSelect(
    const std::vector<T>& population,
    const std::vector<DebProfile>& profiles,
    std::size_t tournamentSize,
    std::mt19937& rng) {
    
    if (population.empty() || profiles.size() != population.size()) {
        return 0;
    }
    std::uniform_int_distribution<std::size_t> dist(0, population.size() - 1);
    std::size_t bestIdx = dist(rng);

    for (std::size_t i = 1; i < tournamentSize; ++i) {
        std::size_t cand = dist(rng);
        if (debIsBetter(profiles[cand], profiles[bestIdx])) {
            bestIdx = cand;
        }
    }
    return bestIdx;
}

} // namespace constraints
} // namespace ga
