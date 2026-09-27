#pragma once

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ga/crossover/base_crossover.h"
#include "ga/metaheuristics/common.hpp"
#include "ga/shake/shake_types.hpp"
#include "ga/shake/shake_operators.hpp"

namespace ga {
namespace hybrid {

struct SwarmGeneticHybridConfig {
    ga::metaheuristics::SearchConfig search;
    double crossoverProbability = 0.7;
    double mutationProbability = 0.15;
    double inertia = 0.7298;
    double cognitive = 1.49618;
    double social = 1.49618;
    std::shared_ptr<CrossoverOperator> crossover;
    ga::shake::ShakeConfig shake;
};

class SwarmGeneticHybridOptimizer final : public ga::metaheuristics::IContinuousOptimizer {
public:
    explicit SwarmGeneticHybridOptimizer(SwarmGeneticHybridConfig config = {})
        : config_(std::move(config)) {}

    std::string name() const override { return "SwarmGeneticHybrid"; }

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

        std::vector<double> gbest(dim);
        double gbestFit = -1e300;

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

        const ga::Bounds bounds = config_.search.bounds;

        ga::core::OptimizationResult res;
        res.bestFitness = gbestFit;
        res.bestSolution = gbest;
        res.generations = 0;
        res.evaluations = popSize;

        std::uniform_real_distribution<double> u01(0.0, 1.0);

        for (std::size_t iter = 0; iter < iters; ++iter) {
            res.generations = iter + 1;
            res.bestHistory.push_back(gbestFit);

            // 1. PSO Step
            for (std::size_t i = 0; i < popSize; ++i) {
                for (std::size_t d = 0; d < dim; ++d) {
                    const double r1 = u01(rng), r2 = u01(rng);
                    vel[i][d] = config_.inertia * vel[i][d]
                              + config_.cognitive * r1 * (pbest[i][d] - pos[i][d])
                              + config_.social * r2 * (gbest[d] - pos[i][d]);
                    pos[i][d] = std::clamp(pos[i][d] + vel[i][d], bounds.lower, bounds.upper);
                }
                const double fit = fitness(pos[i]);
                res.evaluations++;
                if (fit > pbestFit[i]) {
                    pbestFit[i] = fit;
                    pbest[i] = pos[i];
                    if (fit > gbestFit) { gbestFit = fit; gbest = pos[i]; }
                }
            }

            // 2. GA Crossover on pbest
            if (config_.crossover && u01(rng) < config_.crossoverProbability) {
                std::uniform_int_distribution<std::size_t> distIdx(0, popSize - 1);
                std::size_t idx1 = distIdx(rng), idx2 = distIdx(rng);
                auto [c1, c2] = config_.crossover->crossover(pbest[idx1], pbest[idx2]);
                for (std::size_t d = 0; d < dim; ++d) {
                    c1[d] = std::clamp(c1[d], bounds.lower, bounds.upper);
                }
                const double f1 = fitness(c1);
                res.evaluations++;
                if (f1 > pbestFit[idx1]) {
                    pbestFit[idx1] = f1;
                    pbest[idx1] = c1;
                    if (f1 > gbestFit) { gbestFit = f1; gbest = c1; }
                }
            }

            // 3. Shake perturbation
            if (u01(rng) < config_.mutationProbability) {
                std::uniform_int_distribution<std::size_t> distIdx(0, popSize - 1);
                std::size_t target = distIdx(rng);
                std::vector<double> cand = pos[target];
                ga::shake::applyShake(cand, bounds, config_.shake, rng);
                const double f = fitness(cand);
                res.evaluations++;
                if (f > pbestFit[target]) {
                    pbestFit[target] = f;
                    pbest[target] = cand;
                    pos[target] = std::move(cand);
                    if (f > gbestFit) { gbestFit = f; gbest = pbest[target]; }
                }
            }
        }

        res.bestFitness = gbestFit;
        res.bestSolution = gbest;
        return res;
    }

    const SwarmGeneticHybridConfig& config() const noexcept { return config_; }

private:
    SwarmGeneticHybridConfig config_;
};

} // namespace hybrid
} // namespace ga
