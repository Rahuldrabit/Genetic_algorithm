#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ga/config.hpp"
#include "ga/core/result.hpp"
#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace metaheuristics {

enum class CoolingSchedule {
    Geometric,
    Linear,
    Logarithmic
};

struct SimulatedAnnealingConfig {
    SearchConfig search;
    double initialTemperature = 100.0;
    double minTemperature = 1e-5;
    double coolingRate = 0.95; // For geometric cooling
    double stepSize = 0.05;    // Neighborhood perturbation scale relative to bounds
    std::size_t stepsPerTemperature = 1;
    CoolingSchedule schedule = CoolingSchedule::Geometric;
};

class SimulatedAnnealingOptimizer : public IContinuousOptimizer {
public:
    explicit SimulatedAnnealingOptimizer(SimulatedAnnealingConfig config = {})
        : cfg_(std::move(config)) {
        if (cfg_.initialTemperature <= 0.0) cfg_.initialTemperature = 100.0;
        if (cfg_.minTemperature <= 0.0) cfg_.minTemperature = 1e-5;
        if (cfg_.coolingRate <= 0.0 || cfg_.coolingRate >= 1.0) cfg_.coolingRate = 0.95;
        if (cfg_.stepSize <= 0.0) cfg_.stepSize = 0.05;
        if (cfg_.stepsPerTemperature == 0) cfg_.stepsPerTemperature = 1;
    }

    std::string name() const override { return "SimulatedAnnealingOptimizer"; }

    ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const SeedPopulation& seeds = {}) override {
        
        detail::validateSearchConfig(cfg_.search);
        const std::size_t dim = cfg_.search.dimension;
        const auto bounds = cfg_.search.bounds;
        const double boundSpan = bounds.upper - bounds.lower;
        const double deltaRadius = cfg_.stepSize * boundSpan;

        std::mt19937 rng = detail::makeRng(cfg_.search.seed);
        std::uniform_real_distribution<double> uniformProb(0.0, 1.0);
        std::normal_distribution<double> neighborNoise(0.0, deltaRadius);

        // Pre-allocate state vectors (zero heap allocation in inner loop for real-time <1ms execution)
        std::vector<double> currentX(dim);
        std::vector<double> candidateX(dim);
        std::vector<double> bestX(dim);

        if (!seeds.empty() && seeds[0].size() == dim) {
            currentX = seeds[0];
            detail::clampToBounds(currentX, bounds);
        } else {
            std::uniform_real_distribution<double> initDist(bounds.lower, bounds.upper);
            for (std::size_t d = 0; d < dim; ++d) {
                currentX[d] = initDist(rng);
            }
        }

        double currentFitness = fitness(currentX);
        double bestFitness = currentFitness;
        bestX = currentX;

        std::size_t evaluations = 1;
        double temperature = cfg_.initialTemperature;

        ga::core::OptimizationResult res;
        res.bestHistory.reserve(cfg_.search.iterations);
        res.avgHistory.reserve(cfg_.search.iterations);

        for (std::size_t iter = 0; iter < cfg_.search.iterations; ++iter) {
            for (std::size_t step = 0; step < cfg_.stepsPerTemperature; ++step) {
                // Generate neighbor
                for (std::size_t d = 0; d < dim; ++d) {
                    candidateX[d] = std::clamp(currentX[d] + neighborNoise(rng), bounds.lower, bounds.upper);
                }

                double candidateFitness = fitness(candidateX);
                evaluations++;

                double deltaE = candidateFitness - currentFitness; // Maximization

                // Metropolis acceptance criterion
                if (deltaE >= 0.0 || uniformProb(rng) < std::exp(deltaE / std::max(temperature, 1e-12))) {
                    currentX = candidateX;
                    currentFitness = candidateFitness;

                    if (currentFitness > bestFitness) {
                        bestFitness = currentFitness;
                        bestX = currentX;
                    }
                }
            }

            // Cool down
            if (cfg_.schedule == CoolingSchedule::Geometric) {
                temperature = std::max(cfg_.minTemperature, temperature * cfg_.coolingRate);
            } else if (cfg_.schedule == CoolingSchedule::Linear) {
                double decrement = (cfg_.initialTemperature - cfg_.minTemperature) / static_cast<double>(cfg_.search.iterations);
                temperature = std::max(cfg_.minTemperature, temperature - decrement);
            } else {
                temperature = std::max(cfg_.minTemperature, cfg_.initialTemperature / (1.0 + std::log(1.0 + static_cast<double>(iter + 1))));
            }

            res.bestHistory.push_back(bestFitness);
            res.avgHistory.push_back(currentFitness);
        }

        res.bestSolution = std::move(bestX);
        res.bestFitness = bestFitness;
        res.evaluations = evaluations;
        res.generations = cfg_.search.iterations;

        return res;
    }

private:
    SimulatedAnnealingConfig cfg_;
};

} // namespace metaheuristics
} // namespace ga
