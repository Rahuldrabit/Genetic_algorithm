#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

namespace ga {
namespace constraints {

struct AdaptivePenaltyConfig {
    double baseCoefficient = 0.5; // c in (c * t)^alpha
    double alpha = 2.0;           // Power exponent for generation index t
    double beta = 2.0;            // Power exponent for violation degree
    double minPenalty = 1e-4;
    double maxPenalty = 1e9;
};

class AdaptivePenaltyHandler {
public:
    explicit AdaptivePenaltyHandler(AdaptivePenaltyConfig config = {})
        : cfg_(config) {}

    // Computes dynamic penalty factor for generation/iteration t
    double penaltyFactor(std::size_t iteration) const {
        double t = static_cast<double>(iteration + 1);
        double factor = std::pow(cfg_.baseCoefficient * t, cfg_.alpha);
        return std::clamp(factor, cfg_.minPenalty, cfg_.maxPenalty);
    }

    // Applies dynamic penalty: fitness - factor(t) * (violation ^ beta)
    double penalize(double rawFitness, double violation, std::size_t iteration) const {
        if (violation <= 0.0) {
            return rawFitness;
        }
        double factor = penaltyFactor(iteration);
        double penalty = factor * std::pow(violation, cfg_.beta);
        return rawFitness - penalty;
    }

private:
    AdaptivePenaltyConfig cfg_;
};

} // namespace constraints
} // namespace ga
