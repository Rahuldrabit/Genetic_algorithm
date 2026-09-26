#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace pso {

struct ClpsoConfig {
    ga::metaheuristics::SearchConfig search;
    double inertia = 0.7298;
    double learningRate = 1.49618;
    double velocityClamp = 0.2;
    std::size_t refreshGap = 7;
    std::shared_ptr<const ga::metaheuristics::IAdaptiveController> controller;
};

class ComprehensiveLearningPso final : public ga::metaheuristics::IContinuousOptimizer {
public:
    explicit ComprehensiveLearningPso(ClpsoConfig config = {})
        : config_(std::move(config)) {}

    std::string name() const override { return "CLPSO"; }

    ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const ga::metaheuristics::SeedPopulation& seeds = {}) override {
        ga::metaheuristics::detail::validateSearchConfig(config_.search);
        const std::size_t dim = config_.search.dimension;
        const std::size_t popSize = config_.search.populationSize;
        const std::size_t iters = config_.search.iterations;

        std::mt19937 rng(config_.search.seed == 0 ? std::random_device{}() : config_.search.seed);
        std::vector<std::vector<double>> pos(popSize, std::vector<double>(dim));
        std::vector<std::vector<double>> vel(popSize, std::vector<double>(dim, 0.0));
        std::vector<std::vector<double>> pbest(popSize, std::vector<double>(dim));
        std::vector<double> pbestFit(popSize, -1e300);
        std::vector<std::size_t> stagCount(popSize, 0);

        std::vector<double> gbest(dim);
        double gbestFit = -1e300;

        const double span = config_.search.bounds.upper - config_.search.bounds.lower;

        // Learning probabilities Pc
        std::vector<double> pc(popSize);
        for (std::size_t i = 0; i < popSize; ++i) {
            const double ratio = static_cast<double>(i) / static_cast<double>(std::max<std::size_t>(1, popSize - 1));
            pc[i] = 0.05 + 0.45 * (std::exp(10.0 * ratio) - 1.0) / (std::exp(10.0) - 1.0);
        }

        for (std::size_t i = 0; i < popSize; ++i) {
            if (i < seeds.size() && seeds[i].size() == dim) {
                pos[i] = seeds[i];
            } else {
                std::uniform_real_distribution<double> dist(config_.search.bounds.lower, config_.search.bounds.upper);
                for (std::size_t d = 0; d < dim; ++d) {
                    pos[i][d] = dist(rng);
                }
            }
            pbest[i] = pos[i];
            pbestFit[i] = fitness(pos[i]);
            if (pbestFit[i] > gbestFit) { gbestFit = pbestFit[i]; gbest = pos[i]; }
        }

        std::vector<std::vector<std::size_t>> exemplars(popSize, std::vector<std::size_t>(dim));
        auto selectExemplars = [&](std::size_t i) {
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            std::uniform_int_distribution<std::size_t> uIdx(0, popSize - 1);
            for (std::size_t d = 0; d < dim; ++d) {
                if (u01(rng) > pc[i]) {
                    exemplars[i][d] = i;
                } else {
                    std::size_t p1 = uIdx(rng), p2 = uIdx(rng);
                    exemplars[i][d] = (pbestFit[p1] >= pbestFit[p2]) ? p1 : p2;
                }
            }
        };
        for (std::size_t i = 0; i < popSize; ++i) selectExemplars(i);

        ga::core::OptimizationResult res;
        res.bestFitness = gbestFit;
        res.bestSolution = gbest;
        res.generations = 0;
        res.evaluations = popSize;

        for (std::size_t iter = 0; iter < iters; ++iter) {
            res.generations = iter + 1;
            res.bestHistory.push_back(gbestFit);

            double w = config_.inertia - static_cast<double>(iter) / static_cast<double>(iters) * 0.5 * config_.inertia;
            for (std::size_t i = 0; i < popSize; ++i) {
                if (stagCount[i] >= config_.refreshGap) {
                    selectExemplars(i);
                    stagCount[i] = 0;
                }
                std::uniform_real_distribution<double> u01(0.0, 1.0);
                for (std::size_t d = 0; d < dim; ++d) {
                    const std::size_t ex = exemplars[i][d];
                    vel[i][d] = w * vel[i][d] + config_.learningRate * u01(rng) * (pbest[ex][d] - pos[i][d]);
                    const double maxV = config_.velocityClamp * span;
                    vel[i][d] = std::clamp(vel[i][d], -maxV, maxV);
                    pos[i][d] = std::clamp(pos[i][d] + vel[i][d], config_.search.bounds.lower, config_.search.bounds.upper);
                }

                const double fit = fitness(pos[i]);
                res.evaluations++;
                if (fit > pbestFit[i]) {
                    pbestFit[i] = fit;
                    pbest[i] = pos[i];
                    stagCount[i] = 0;
                    if (fit > gbestFit) { gbestFit = fit; gbest = pos[i]; }
                } else {
                    stagCount[i]++;
                }
            }
        }
        res.bestFitness = gbestFit;
        res.bestSolution = gbest;
        return res;
    }

    const ClpsoConfig& config() const noexcept { return config_; }

private:
    ClpsoConfig config_;
};

} // namespace pso
} // namespace ga
