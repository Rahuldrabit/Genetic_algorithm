#pragma once

#include <functional>
#include <numeric>
#include <vector>

#include "ga/es/cmaes.hpp"

namespace ga {
namespace moea {

struct MoCmaEsConfig {
    ga::es::CmaEsConfig cma;
    std::vector<double> weights;
};

struct MoCmaEsResult {
    std::vector<double> best;
    std::vector<double> objectives;
    double weightedFitness = -1e300;
};

class MoCmaEs {
public:
    using ObjectiveFn = std::function<std::vector<double>(const std::vector<double>&)>;

    explicit MoCmaEs(MoCmaEsConfig cfg)
        : cfg_(std::move(cfg)) {}

    MoCmaEsResult run(const ObjectiveFn& objectiveFn) const {
        ga::es::DiagonalCmaEs cma(cfg_.cma);

        auto scalarized = [&](const std::vector<double>& x) {
            const auto obj = objectiveFn(x);
            const std::size_t n = std::min(obj.size(), cfg_.weights.size());
            if (n == 0) {
                return 0.0;
            }
            double value = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                value += cfg_.weights[i] * obj[i];
            }
            // convert minimization weighted sum to maximization for CMA wrapper
            return -value;
        };

        const auto base = cma.run(scalarized);
        MoCmaEsResult out;
        out.best = base.best;
        out.weightedFitness = base.bestFitness;
        out.objectives = objectiveFn(base.best);
        return out;
    }

private:
    MoCmaEsConfig cfg_;
};

} // namespace moea
} // namespace ga
