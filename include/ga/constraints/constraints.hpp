#pragma once

#include <functional>
#include <vector>

namespace ga {
namespace constraints {

using Constraint = std::function<bool(const std::vector<double>&)>;
using PenaltyFn = std::function<double(const std::vector<double>&)>;
using RepairFn = std::function<void(std::vector<double>&)>;

struct ConstraintSet {
    std::vector<Constraint> hard;
    std::vector<PenaltyFn> soft;
    std::vector<RepairFn> repairs;
};

inline bool isFeasible(const std::vector<double>& genes, const ConstraintSet& set) {
    for (const auto& c : set.hard) {
        if (c && !c(genes)) {
            return false;
        }
    }
    return true;
}

inline double totalPenalty(const std::vector<double>& genes, const ConstraintSet& set) {
    double penalty = 0.0;
    for (const auto& p : set.soft) {
        if (p) {
            penalty += p(genes);
        }
    }
    return penalty;
}

inline void applyRepairs(std::vector<double>& genes, const ConstraintSet& set) {
    for (const auto& r : set.repairs) {
        if (r) {
            r(genes);
        }
    }
}

inline double penalizedFitness(double baseFitness,
                               const std::vector<double>& genes,
                               const ConstraintSet& set,
                               double infeasiblePenalty = 1e6) {
    double fitness = baseFitness;
    if (!isFeasible(genes, set)) {
        fitness -= infeasiblePenalty;
    }
    fitness -= totalPenalty(genes, set);
    return fitness;
}

} // namespace constraints
} // namespace ga
