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

struct HillClimbingConfig {
    SearchConfig search;
    double stepSize = 0.02; // Perturbation step as fraction of bounds span
    std::size_t numNeighbors = 5; // Candidate neighbors sampled per step (steepest ascent)
    std::size_t restarts = 0; // Number of random restarts to avoid local traps
    bool steepestAscent = true; // True for best neighbor, false for first improving
};

class HillClimbingOptimizer : public IContinuousOptimizer {
public:
    explicit HillClimbingOptimizer(HillClimbingConfig config = {})
        : cfg_(std::move(config)) {
        if (cfg_.stepSize <= 0.0) cfg_.stepSize = 0.02;
        if (cfg_.numNeighbors == 0) cfg_.numNeighbors = 1;
    }

    std::string name() const override { return "HillClimbingOptimizer"; }

    ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const SeedPopulation& seeds = {}) override {
        
        detail::validateSearchConfig(cfg_.search);
        const std::size_t dim = cfg_.search.dimension;
        const auto bounds = cfg_.search.bounds;
        const double boundSpan = bounds.upper - bounds.lower;
        const double radius = cfg_.stepSize * boundSpan;

        std::mt19937 rng = detail::makeRng(cfg_.search.seed);
        std::normal_distribution<double> stepNoise(0.0, radius);
        std::uniform_real_distribution<double> initDist(bounds.lower, bounds.upper);

        ga::core::OptimizationResult res;
        res.bestFitness = -std::numeric_limits<double>::infinity();
        res.bestSolution.resize(dim);
        res.bestHistory.reserve(cfg_.search.iterations);
        res.avgHistory.reserve(cfg_.search.iterations);

        std::size_t totalEvaluations = 0;
        const std::size_t totalRestarts = cfg_.restarts;
        const std::size_t itersPerRun = cfg_.search.iterations / (totalRestarts + 1);

        std::vector<double> currentX(dim);
        std::vector<double> neighborX(dim);
        std::vector<double> bestNeighborX(dim);

        for (std::size_t run = 0; run <= totalRestarts; ++run) {
            // Initialize starting point
            if (run == 0 && !seeds.empty() && seeds[0].size() == dim) {
                currentX = seeds[0];
                detail::clampToBounds(currentX, bounds);
            } else {
                for (std::size_t d = 0; d < dim; ++d) {
                    currentX[d] = initDist(rng);
                }
            }

            double currentFitness = fitness(currentX);
            totalEvaluations++;

            if (currentFitness > res.bestFitness) {
                res.bestFitness = currentFitness;
                res.bestSolution = currentX;
            }

            for (std::size_t it = 0; it < itersPerRun; ++it) {
                bool improved = false;
                double bestNeighborFit = -std::numeric_limits<double>::infinity();

                for (std::size_t n = 0; n < cfg_.numNeighbors; ++n) {
                    for (std::size_t d = 0; d < dim; ++d) {
                        neighborX[d] = std::clamp(currentX[d] + stepNoise(rng), bounds.lower, bounds.upper);
                    }
                    double neighborFit = fitness(neighborX);
                    totalEvaluations++;

                    if (neighborFit > currentFitness) {
                        improved = true;
                        if (!cfg_.steepestAscent) {
                            currentX = neighborX;
                            currentFitness = neighborFit;
                            break;
                        }
                    }

                    if (neighborFit > bestNeighborFit) {
                        bestNeighborFit = neighborFit;
                        bestNeighborX = neighborX;
                    }
                }

                if (cfg_.steepestAscent && improved && bestNeighborFit > currentFitness) {
                    currentX = bestNeighborX;
                    currentFitness = bestNeighborFit;
                }

                if (currentFitness > res.bestFitness) {
                    res.bestFitness = currentFitness;
                    res.bestSolution = currentX;
                }

                res.bestHistory.push_back(res.bestFitness);
                res.avgHistory.push_back(currentFitness);
            }
        }

        res.evaluations = totalEvaluations;
        res.generations = cfg_.search.iterations;
        return res;
    }

private:
    HillClimbingConfig cfg_;
};

} // namespace metaheuristics
} // namespace ga
