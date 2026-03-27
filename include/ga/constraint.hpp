#pragma once
/// Constraint Handling
///
/// Three strategies:
///   1. DeathPenalty   – discard (assign -∞) any infeasible solution.
///   2. PenaltyMethod  – subtract a weighted penalty from the raw fitness.
///   3. RepairOperator – project infeasible genes back into the feasible space.
///
/// All constraint predicates follow the convention:
///   constraint(genes) returns true  when the constraint IS SATISFIED.
///   constraint(genes) returns false when the constraint IS VIOLATED.

#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <limits>

namespace ga {
namespace constraint {

// ============================================================================
// Constraint – one named predicate
// ============================================================================

struct Constraint {
    std::string name;
    /// Returns true if genes satisfy this constraint.
    std::function<bool(const std::vector<double>&)> isSatisfied;
    /// Returns non-negative violation magnitude (0 when satisfied).
    std::function<double(const std::vector<double>&)> violation;
};

// ============================================================================
// Helper factories
// ============================================================================

/// Bound constraint: lowerBound[i] <= genes[i] <= upperBound[i].
inline Constraint boundConstraint(std::vector<double> lb, std::vector<double> ub) {
    return {
        "bounds",
        [lb, ub](const std::vector<double>& g) {
            for (size_t i = 0; i < g.size(); ++i)
                if (g[i] < lb[i] || g[i] > ub[i]) return false;
            return true;
        },
        [lb, ub](const std::vector<double>& g) {
            double v = 0.0;
            for (size_t i = 0; i < g.size(); ++i) {
                if (g[i] < lb[i]) v += lb[i] - g[i];
                if (g[i] > ub[i]) v += g[i] - ub[i];
            }
            return v;
        }
    };
}

/// Linear inequality: sum(coeffs * genes) <= rhs.
inline Constraint linearIneq(std::vector<double> coeffs, double rhs) {
    return {
        "linear_ineq",
        [coeffs, rhs](const std::vector<double>& g) {
            double s = 0.0;
            for (size_t i = 0; i < g.size() && i < coeffs.size(); ++i)
                s += coeffs[i] * g[i];
            return s <= rhs;
        },
        [coeffs, rhs](const std::vector<double>& g) {
            double s = 0.0;
            for (size_t i = 0; i < g.size() && i < coeffs.size(); ++i)
                s += coeffs[i] * g[i];
            return std::max(0.0, s - rhs);
        }
    };
}

// ============================================================================
// ConstraintHandler – applies one of the three strategies
// ============================================================================

class ConstraintHandler {
public:
    explicit ConstraintHandler(std::vector<Constraint> constraints,
                               double penaltyWeight = 1000.0)
        : constraints_(std::move(constraints)),
          penaltyWeight_(penaltyWeight) {}

    /// Death-penalty: returns -inf if any constraint is violated.
    double deathPenalty(const std::vector<double>& genes, double rawFitness) const {
        for (const auto& c : constraints_)
            if (!c.isSatisfied(genes))
                return -std::numeric_limits<double>::infinity();
        return rawFitness;
    }

    /// Penalty method: subtract penaltyWeight * total_violation.
    double penaltyFitness(const std::vector<double>& genes, double rawFitness) const {
        double penalty = 0.0;
        for (const auto& c : constraints_)
            penalty += c.violation(genes);
        return rawFitness - penaltyWeight_ * penalty;
    }

    /// Returns total violation magnitude.
    double totalViolation(const std::vector<double>& genes) const {
        double v = 0.0;
        for (const auto& c : constraints_) v += c.violation(genes);
        return v;
    }

    bool isFeasible(const std::vector<double>& genes) const {
        for (const auto& c : constraints_)
            if (!c.isSatisfied(genes)) return false;
        return true;
    }

private:
    std::vector<Constraint> constraints_;
    double penaltyWeight_;
};

// ============================================================================
// BoundRepair – projects genes back into [lb, ub] (simplest repair op)
// ============================================================================

class BoundRepair {
public:
    explicit BoundRepair(std::vector<double> lb, std::vector<double> ub)
        : lb_(std::move(lb)), ub_(std::move(ub)) {}

    void repair(std::vector<double>& genes) const {
        for (size_t i = 0; i < genes.size(); ++i) {
            if (i < lb_.size()) genes[i] = std::max(lb_[i], genes[i]);
            if (i < ub_.size()) genes[i] = std::min(ub_[i], genes[i]);
        }
    }

private:
    std::vector<double> lb_, ub_;
};

} // namespace constraint
} // namespace ga
