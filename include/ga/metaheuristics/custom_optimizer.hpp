#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace metaheuristics {

using CustomAlgorithmFunction = std::function<ga::core::OptimizationResult(
    const ga::Fitness& fitness,
    const SearchConfig& config,
    const SeedPopulation& seeds,
    const IAdaptiveController* controller)>;

class CustomContinuousOptimizer final : public IContinuousOptimizer {
public:
    CustomContinuousOptimizer(std::string name,
                              SearchConfig config,
                              CustomAlgorithmFunction function,
                              std::shared_ptr<const IAdaptiveController> controller = nullptr)
        : name_(std::move(name)),
          config_(config),
          function_(std::move(function)),
          controller_(std::move(controller)) {
        if (!function_) {
            throw std::invalid_argument("CustomAlgorithmFunction cannot be empty");
        }
        detail::validateSearchConfig(config_);
    }

    std::string name() const override { return name_; }

    ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const SeedPopulation& seeds = {}) override {
        return function_(fitness, config_, seeds, controller_.get());
    }

    const SearchConfig& config() const noexcept { return config_; }
    void setConfig(SearchConfig config) {
        detail::validateSearchConfig(config);
        config_ = config;
    }

    void setController(std::shared_ptr<const IAdaptiveController> controller) noexcept {
        controller_ = std::move(controller);
    }

private:
    std::string name_;
    SearchConfig config_;
    CustomAlgorithmFunction function_;
    std::shared_ptr<const IAdaptiveController> controller_;
};

inline std::shared_ptr<CustomContinuousOptimizer> makeCustomOptimizer(
    std::string name,
    SearchConfig config,
    CustomAlgorithmFunction function,
    std::shared_ptr<const IAdaptiveController> controller = nullptr) {
    return std::make_shared<CustomContinuousOptimizer>(
        std::move(name), config, std::move(function), std::move(controller));
}

} // namespace metaheuristics
} // namespace ga
