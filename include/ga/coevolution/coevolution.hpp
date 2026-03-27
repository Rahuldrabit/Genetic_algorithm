#pragma once

#include <functional>
#include <random>
#include <utility>
#include <vector>

#include "ga/core/individual.hpp"

namespace ga {
namespace coevolution {

struct CoevolutionConfig {
    std::size_t generations = 50;
    unsigned seed = 0;
};

class CoevolutionEngine {
public:
    using Populations = std::vector<std::vector<ga::Individual>>;
    using EvaluateFn = std::function<void(Populations&)>;
    using ReproduceFn = std::function<void(Populations&, std::mt19937&)>;

    explicit CoevolutionEngine(CoevolutionConfig cfg)
        : cfg_(std::move(cfg)) {}

    Populations run(Populations populations,
                    const EvaluateFn& evaluate,
                    const ReproduceFn& reproduce) const {
        std::mt19937 rng(cfg_.seed == 0 ? std::random_device{}() : cfg_.seed);

        if (evaluate) {
            evaluate(populations);
        }
        for (std::size_t gen = 0; gen < cfg_.generations; ++gen) {
            if (reproduce) {
                reproduce(populations, rng);
            }
            if (evaluate) {
                evaluate(populations);
            }
        }
        return populations;
    }

private:
    CoevolutionConfig cfg_;
};

} // namespace coevolution
} // namespace ga
