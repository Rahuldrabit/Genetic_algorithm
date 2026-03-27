#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

namespace ga {
namespace es {

struct EvolutionStrategyConfig {
    std::size_t mu = 20;
    std::size_t lambda = 100;
    std::size_t generations = 100;
    std::size_t dimension = 10;
    double sigma = 0.3;
    double lower = -5.0;
    double upper = 5.0;
    bool plusStrategy = true; // true=(mu+lambda), false=(mu,lambda)
    unsigned seed = 0;
};

struct EvolutionStrategyResult {
    std::vector<double> best;
    double bestFitness = -1e300;
    std::vector<double> bestHistory;
};

class EvolutionStrategy {
public:
    using Fitness = std::function<double(const std::vector<double>&)>;

    explicit EvolutionStrategy(EvolutionStrategyConfig cfg)
        : cfg_(std::move(cfg)) {}

    EvolutionStrategyResult run(const Fitness& fitness) const {
        std::mt19937 rng(cfg_.seed == 0 ? std::random_device{}() : cfg_.seed);
        std::uniform_real_distribution<double> init(cfg_.lower, cfg_.upper);
        std::normal_distribution<double> noise(0.0, cfg_.sigma);

        struct Candidate {
            std::vector<double> genes;
            double fitness = -1e300;
        };

        auto eval = [&](Candidate& c) { c.fitness = fitness(c.genes); };

        std::vector<Candidate> parents(cfg_.mu);
        for (auto& p : parents) {
            p.genes.resize(cfg_.dimension);
            for (double& g : p.genes) {
                g = init(rng);
            }
            eval(p);
        }

        EvolutionStrategyResult result;
        result.bestHistory.reserve(cfg_.generations + 1);

        auto updateBest = [&](const std::vector<Candidate>& pool) {
            auto it = std::max_element(pool.begin(), pool.end(), [](const Candidate& a, const Candidate& b) {
                return a.fitness < b.fitness;
            });
            if (it != pool.end() && it->fitness > result.bestFitness) {
                result.bestFitness = it->fitness;
                result.best = it->genes;
            }
            result.bestHistory.push_back(result.bestFitness);
        };

        updateBest(parents);

        for (std::size_t gen = 0; gen < cfg_.generations; ++gen) {
            std::vector<Candidate> offspring;
            offspring.reserve(cfg_.lambda);
            std::uniform_int_distribution<std::size_t> pick(0, parents.size() - 1);

            for (std::size_t i = 0; i < cfg_.lambda; ++i) {
                Candidate child = parents[pick(rng)];
                for (double& g : child.genes) {
                    g = std::clamp(g + noise(rng), cfg_.lower, cfg_.upper);
                }
                eval(child);
                offspring.push_back(std::move(child));
            }

            std::vector<Candidate> candidates;
            if (cfg_.plusStrategy) {
                candidates = parents;
                candidates.insert(candidates.end(), offspring.begin(), offspring.end());
            } else {
                candidates = std::move(offspring);
            }

            std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
                return a.fitness > b.fitness;
            });
            candidates.resize(cfg_.mu);
            parents = std::move(candidates);

            updateBest(parents);
        }

        return result;
    }

private:
    EvolutionStrategyConfig cfg_;
};

} // namespace es
} // namespace ga
