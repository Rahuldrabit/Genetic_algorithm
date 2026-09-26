#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace hybrid {

struct InterleavedHybridConfig {
    ga::metaheuristics::SearchConfig search;
    std::size_t epochs = 3;
    std::size_t stage1Iterations = 15;
    std::size_t stage2Iterations = 15;
    std::size_t migratedEliteCount = 5;
};

class InterleavedHybridOptimizer final : public ga::metaheuristics::IContinuousOptimizer {
public:
    InterleavedHybridOptimizer(
        std::shared_ptr<ga::metaheuristics::IContinuousOptimizer> stage1,
        std::shared_ptr<ga::metaheuristics::IContinuousOptimizer> stage2,
        InterleavedHybridConfig config = {})
        : stage1_(std::move(stage1)), stage2_(std::move(stage2)), config_(std::move(config)) {}

    std::string name() const override {
        const std::string n1 = stage1_ ? stage1_->name() : "None";
        const std::string n2 = stage2_ ? stage2_->name() : "None";
        return "InterleavedHybrid[" + n1 + "+" + n2 + "]";
    }

    ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const ga::metaheuristics::SeedPopulation& seeds = {}) override {
        if (!stage1_ || !stage2_) {
            throw std::invalid_argument("InterleavedHybridOptimizer requires non-null stage optimizers");
        }

        ga::core::OptimizationResult result;
        result.bestFitness = -1e300;
        result.generations = 0;
        result.evaluations = 0;

        std::vector<std::vector<double>> currentSeeds = seeds;

        for (std::size_t epoch = 0; epoch < config_.epochs; ++epoch) {
            auto r1 = stage1_->optimize(fitness, currentSeeds);
            result.generations += r1.generations;
            result.evaluations += r1.evaluations;
            for (double f : r1.bestHistory) result.bestHistory.push_back(f);
            if (r1.bestFitness > result.bestFitness) {
                result.bestFitness = r1.bestFitness;
                result.bestSolution = r1.bestSolution;
            }

            currentSeeds.clear();
            if (!result.bestSolution.empty()) currentSeeds.push_back(result.bestSolution);

            auto r2 = stage2_->optimize(fitness, currentSeeds);
            result.generations += r2.generations;
            result.evaluations += r2.evaluations;
            for (double f : r2.bestHistory) result.bestHistory.push_back(f);
            if (r2.bestFitness > result.bestFitness) {
                result.bestFitness = r2.bestFitness;
                result.bestSolution = r2.bestSolution;
            }

            currentSeeds.clear();
            if (!result.bestSolution.empty()) currentSeeds.push_back(result.bestSolution);
        }

        return result;
    }

    const InterleavedHybridConfig& config() const noexcept { return config_; }

private:
    std::shared_ptr<ga::metaheuristics::IContinuousOptimizer> stage1_;
    std::shared_ptr<ga::metaheuristics::IContinuousOptimizer> stage2_;
    InterleavedHybridConfig config_;
};

} // namespace hybrid
} // namespace ga
