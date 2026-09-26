#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ga/gso/gso_types.hpp"
#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace gso {

class GlowwormSwarmOptimizer final : public ga::metaheuristics::IContinuousOptimizer {
public:
    explicit GlowwormSwarmOptimizer(GsoConfig config = {})
        : config_(std::move(config)) {}

    std::string name() const override {
        switch (config_.variant) {
            case GsoVariant::AdaptiveStep: return "GSO-AdaptiveStep";
            case GsoVariant::LevyFlight: return "GSO-LevyFlight";
            case GsoVariant::MultiModal: return "GSO-MultiModal";
            default: return "GSO-Standard";
        }
    }

    ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const ga::metaheuristics::SeedPopulation& seeds = {}) override {
        ga::metaheuristics::detail::validateSearchConfig(config_.search);
        const std::size_t dim = config_.search.dimension;
        const std::size_t popSize = config_.search.populationSize;
        const std::size_t iters = config_.search.iterations;

        std::mt19937 rng(config_.search.seed == 0 ? std::random_device{}() : config_.search.seed);
        std::vector<Glowworm> swarm(popSize);

        const double maxSpan = config_.search.bounds.upper - config_.search.bounds.lower;
        const double rSensor = config_.sensorRange * maxSpan;

        for (std::size_t i = 0; i < popSize; ++i) {
            swarm[i].position.resize(dim);
            if (i < seeds.size() && seeds[i].size() == dim) {
                swarm[i].position = seeds[i];
            } else {
                std::uniform_real_distribution<double> dist(config_.search.bounds.lower, config_.search.bounds.upper);
                for (std::size_t d = 0; d < dim; ++d) {
                    swarm[i].position[d] = dist(rng);
                }
            }
            swarm[i].sensorRadius = config_.initialSensorRange * maxSpan;
            swarm[i].luciferin = config_.initialLuciferin;
            swarm[i].fitness = fitness(swarm[i].position);
        }

        std::vector<double> bestPos = swarm[0].position;
        double bestFit = swarm[0].fitness;
        for (const auto& gw : swarm) {
            if (gw.fitness > bestFit) { bestFit = gw.fitness; bestPos = gw.position; }
        }

        ga::core::OptimizationResult result;
        result.bestFitness = bestFit;
        result.bestSolution = bestPos;
        result.generations = 0;
        result.evaluations = popSize;
        result.bestHistory.push_back(bestFit);

        for (std::size_t iter = 0; iter < iters; ++iter) {
            result.generations = iter + 1;
            result.bestHistory.push_back(bestFit);

            // 1. Luciferin update
            for (auto& gw : swarm) {
                gw.fitness = fitness(gw.position);
                result.evaluations++;
                if (gw.fitness > bestFit) { bestFit = gw.fitness; bestPos = gw.position; }
                gw.luciferin = (1.0 - config_.rho) * gw.luciferin + config_.gamma * gw.fitness;
            }

            // 2. Neighbor movement
            std::vector<std::vector<double>> newPositions(popSize);
            for (std::size_t i = 0; i < popSize; ++i) {
                std::vector<std::size_t> neighbors;
                std::vector<double> probs;
                double probSum = 0.0;

                for (std::size_t j = 0; j < popSize; ++j) {
                    if (i == j) continue;
                    double distSq = 0.0;
                    for (std::size_t d = 0; d < dim; ++d) {
                        const double diff = swarm[j].position[d] - swarm[i].position[d];
                        distSq += diff * diff;
                    }
                    const double dist = std::sqrt(distSq);
                    if (dist <= swarm[i].sensorRadius && swarm[j].luciferin > swarm[i].luciferin) {
                        const double diffL = swarm[j].luciferin - swarm[i].luciferin;
                        neighbors.push_back(j);
                        probs.push_back(diffL);
                        probSum += diffL;
                    }
                }

                newPositions[i] = swarm[i].position;
                if (!neighbors.empty() && probSum > 1e-12) {
                    std::uniform_real_distribution<double> u(0.0, probSum);
                    double pick = u(rng), cum = 0.0;
                    std::size_t targetIdx = neighbors.front();
                    for (std::size_t k = 0; k < neighbors.size(); ++k) {
                        cum += probs[k];
                        if (pick <= cum) { targetIdx = neighbors[k]; break; }
                    }

                    double step = config_.stepSize * maxSpan;
                    if (config_.variant == GsoVariant::AdaptiveStep) {
                        const double rel = static_cast<double>(iter) / static_cast<double>(iters);
                        step = (config_.maxStepSize - rel * (config_.maxStepSize - config_.minStepSize)) * maxSpan;
                    }

                    double distToTarget = 0.0;
                    for (std::size_t d = 0; d < dim; ++d) {
                        const double diff = swarm[targetIdx].position[d] - swarm[i].position[d];
                        distToTarget += diff * diff;
                    }
                    distToTarget = std::sqrt(distToTarget);

                    if (distToTarget > 1e-12) {
                        for (std::size_t d = 0; d < dim; ++d) {
                            newPositions[i][d] += step * (swarm[targetIdx].position[d] - swarm[i].position[d]) / distToTarget;
                        }
                    }
                }

                // 3. Sensor range update
                swarm[i].sensorRadius = std::min(rSensor, std::max(0.01 * maxSpan,
                    swarm[i].sensorRadius + config_.beta * (static_cast<double>(config_.targetNeighbors) - static_cast<double>(neighbors.size()))));
            }

            for (std::size_t i = 0; i < popSize; ++i) {
                for (std::size_t d = 0; d < dim; ++d) {
                    swarm[i].position[d] = std::clamp(newPositions[i][d], config_.search.bounds.lower, config_.search.bounds.upper);
                }
            }
        }

        result.bestFitness = bestFit;
        result.bestSolution = bestPos;
        return result;
    }

    const GsoConfig& config() const noexcept { return config_; }

private:
    GsoConfig config_;
};

} // namespace gso
} // namespace ga
