#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ga/api/optimizer.hpp"
#include "ga/config.hpp"

namespace ga {
namespace api {

// Fluent builder for constructing and running an Optimizer.
//
// Usage example:
//   auto result = ga::api::OptimizerBuilder()
//       .dimension(10)
//       .bounds(-5.0, 5.0)
//       .populationSize(100)
//       .generations(200)
//       .seed(42)
//       .threads(4)
//       .build()
//       .optimize(myFitness);
class OptimizerBuilder {
public:
    OptimizerBuilder& dimension(int dim) {
        cfg_.dimension = dim;
        return *this;
    }

    OptimizerBuilder& bounds(double lower, double upper) {
        if (lower >= upper) {
            throw std::invalid_argument("OptimizerBuilder: lower bound must be less than upper bound");
        }
        cfg_.bounds = {lower, upper};
        return *this;
    }

    OptimizerBuilder& populationSize(int size) {
        if (size <= 0) {
            throw std::invalid_argument("OptimizerBuilder: populationSize must be positive");
        }
        cfg_.populationSize = size;
        return *this;
    }

    OptimizerBuilder& generations(int gen) {
        if (gen <= 0) {
            throw std::invalid_argument("OptimizerBuilder: generations must be positive");
        }
        cfg_.generations = gen;
        return *this;
    }

    OptimizerBuilder& seed(unsigned s) {
        cfg_.seed = s;
        return *this;
    }

    OptimizerBuilder& crossoverRate(double rate) {
        if (rate < 0.0 || rate > 1.0) {
            throw std::invalid_argument("OptimizerBuilder: crossoverRate must be in [0, 1]");
        }
        cfg_.crossoverRate = rate;
        return *this;
    }

    OptimizerBuilder& mutationRate(double rate) {
        if (rate < 0.0 || rate > 1.0) {
            throw std::invalid_argument("OptimizerBuilder: mutationRate must be in [0, 1]");
        }
        cfg_.mutationRate = rate;
        return *this;
    }

    OptimizerBuilder& eliteRatio(double ratio) {
        if (ratio < 0.0 || ratio > 1.0) {
            throw std::invalid_argument("OptimizerBuilder: eliteRatio must be in [0, 1]");
        }
        cfg_.eliteRatio = ratio;
        return *this;
    }

    OptimizerBuilder& threads(std::size_t n) {
        threads_ = n;
        return *this;
    }

    // Build and return the configured Optimizer.
    Optimizer build() const {
        Optimizer opt;
        opt.withConfig(cfg_).withThreads(threads_);
        return opt;
    }

private:
    ga::Config cfg_{};
    std::size_t threads_ = 1;
};

} // namespace api
} // namespace ga
