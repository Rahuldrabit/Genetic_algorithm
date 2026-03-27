#pragma once

#include <algorithm>

namespace ga {
namespace adaptive {

struct AdaptiveRates {
    double mutationRate = 0.1;
    double crossoverRate = 0.8;
};

class AdaptiveRateController {
public:
    AdaptiveRateController(double minMutation = 0.001,
                           double maxMutation = 0.6,
                           double minCrossover = 0.4,
                           double maxCrossover = 0.95)
        : minMutation_(minMutation),
          maxMutation_(maxMutation),
          minCrossover_(minCrossover),
          maxCrossover_(maxCrossover) {}

    AdaptiveRates update(const AdaptiveRates& current,
                         double diversity,
                         double bestImprovement) const {
        AdaptiveRates next = current;

        if (diversity < 0.15 || bestImprovement < 1e-9) {
            next.mutationRate = std::min(maxMutation_, current.mutationRate * 1.15 + 0.002);
            next.crossoverRate = std::max(minCrossover_, current.crossoverRate * 0.98);
        } else {
            next.mutationRate = std::max(minMutation_, current.mutationRate * 0.97);
            next.crossoverRate = std::min(maxCrossover_, current.crossoverRate * 1.01);
        }

        return next;
    }

private:
    double minMutation_;
    double maxMutation_;
    double minCrossover_;
    double maxCrossover_;
};

} // namespace adaptive
} // namespace ga
