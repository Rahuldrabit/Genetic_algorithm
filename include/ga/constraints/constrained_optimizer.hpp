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
#include "ga/constraints/constraints.hpp"
#include "ga/constraints/deb_feasibility.hpp"
#include "ga/constraints/adaptive_penalty.hpp"
#include "ga/constraints/repair_operators.hpp"

namespace ga {
namespace constraints {

using ViolationEvaluator = std::function<double(const std::vector<double>&)>;

struct ConstrainedOptimizerConfig {
    ga::metaheuristics::SearchConfig search;
    bool useDebFeasibility = true;
    bool useAdaptivePenalty = true;
    AdaptivePenaltyConfig penaltyConfig;
    double crossoverRate = 0.8;
    double mutationRate = 0.1;
    std::size_t tournamentSize = 3;
};

class ConstrainedOptimizer : public ga::metaheuristics::IContinuousOptimizer {
public:
    explicit ConstrainedOptimizer(
        ConstrainedOptimizerConfig config = {},
        ConstraintSet constraintSet = {},
        ViolationEvaluator violationFn = nullptr)
        : cfg_(std::move(config)),
          constraints_(std::move(constraintSet)),
          violationFn_(std::move(violationFn)),
          penaltyHandler_(cfg_.penaltyConfig) {}

    std::string name() const override { return "ConstrainedOptimizer"; }

    void setViolationEvaluator(ViolationEvaluator fn) {
        violationFn_ = std::move(fn);
    }

    void setConstraintSet(ConstraintSet set) {
        constraints_ = std::move(set);
    }

    ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const ga::metaheuristics::SeedPopulation& seeds = {}) override {
        
        ga::metaheuristics::detail::validateSearchConfig(cfg_.search);
        const std::size_t popSize = cfg_.search.populationSize;
        const std::size_t dim = cfg_.search.dimension;
        const auto bounds = cfg_.search.bounds;
        const double boundSpan = bounds.upper - bounds.lower;

        std::mt19937 rng = ga::metaheuristics::detail::makeRng(cfg_.search.seed);
        std::uniform_real_distribution<double> uniformProb(0.0, 1.0);
        std::uniform_real_distribution<double> initDist(bounds.lower, bounds.upper);
        std::normal_distribution<double> mutNoise(0.0, 0.1 * boundSpan);

        struct Candidate {
            std::vector<double> genes;
            double rawFitness = -std::numeric_limits<double>::infinity();
            double penalizedFitness = -std::numeric_limits<double>::infinity();
            double violation = 0.0;
            bool isFeasible = true;

            DebProfile toDebProfile() const {
                return {rawFitness, violation, isFeasible};
            }
        };

        auto evaluateCandidate = [&](Candidate& cand, std::size_t iteration) {
            // Apply registered repairs first
            applyRepairs(cand.genes, constraints_);
            repairBoxBounds(cand.genes, bounds.lower, bounds.upper);

            // Compute violation
            double viol = 0.0;
            if (!isFeasible(cand.genes, constraints_)) {
                viol += 1.0;
            }
            viol += totalPenalty(cand.genes, constraints_);
            if (violationFn_) {
                viol += violationFn_(cand.genes);
            }

            cand.violation = viol;
            cand.isFeasible = (viol <= 1e-9);
            cand.rawFitness = fitness(cand.genes);

            if (cfg_.useAdaptivePenalty) {
                cand.penalizedFitness = penaltyHandler_.penalize(cand.rawFitness, cand.violation, iteration);
            } else {
                cand.penalizedFitness = cand.rawFitness - (cand.isFeasible ? 0.0 : 1e6 * (1.0 + cand.violation));
            }
        };

        std::vector<Candidate> pop(popSize);
        std::size_t evaluations = 0;

        for (std::size_t i = 0; i < popSize; ++i) {
            pop[i].genes.resize(dim);
            if (i < seeds.size() && seeds[i].size() == dim) {
                pop[i].genes = seeds[i];
            } else {
                for (std::size_t d = 0; d < dim; ++d) {
                    pop[i].genes[d] = initDist(rng);
                }
            }
            evaluateCandidate(pop[i], 0);
            evaluations++;
        }

        ga::core::OptimizationResult res;
        res.bestFitness = -std::numeric_limits<double>::infinity();
        res.bestSolution.resize(dim);
        res.bestHistory.reserve(cfg_.search.iterations);
        res.avgHistory.reserve(cfg_.search.iterations);

        Candidate bestFeasible;
        bestFeasible.isFeasible = false;
        bestFeasible.rawFitness = -std::numeric_limits<double>::infinity();

        auto updateBest = [&]() {
            for (const auto& cand : pop) {
                if (cand.isFeasible) {
                    if (!bestFeasible.isFeasible || cand.rawFitness > bestFeasible.rawFitness) {
                        bestFeasible = cand;
                    }
                }
            }
            if (bestFeasible.isFeasible) {
                res.bestFitness = bestFeasible.rawFitness;
                res.bestSolution = bestFeasible.genes;
            } else {
                // If no feasible found yet, pick candidate with minimal violation
                auto minViol = std::min_element(pop.begin(), pop.end(),
                    [](const Candidate& a, const Candidate& b) { return a.violation < b.violation; });
                res.bestFitness = minViol->rawFitness;
                res.bestSolution = minViol->genes;
            }
        };

        updateBest();

        for (std::size_t iter = 0; iter < cfg_.search.iterations; ++iter) {
            std::vector<DebProfile> debProfiles;
            if (cfg_.useDebFeasibility) {
                debProfiles.reserve(popSize);
                for (const auto& cand : pop) {
                    debProfiles.push_back(cand.toDebProfile());
                }
            }

            auto selectParent = [&](std::mt19937& r) -> const Candidate& {
                if (cfg_.useDebFeasibility) {
                    std::size_t idx = debTournamentSelect(pop, debProfiles, cfg_.tournamentSize, r);
                    return pop[idx];
                }
                std::uniform_int_distribution<std::size_t> dist(0, popSize - 1);
                std::size_t a = dist(r);
                std::size_t b = dist(r);
                return pop[a].penalizedFitness > pop[b].penalizedFitness ? pop[a] : pop[b];
            };

            std::vector<Candidate> nextPop;
            nextPop.reserve(popSize);

            // Elitism: carry over best
            if (bestFeasible.isFeasible) {
                nextPop.push_back(bestFeasible);
            } else {
                nextPop.push_back(pop[0]);
            }

            while (nextPop.size() < popSize) {
                const auto& p1 = selectParent(rng);
                const auto& p2 = selectParent(rng);

                Candidate c1, c2;
                c1.genes.resize(dim);
                c2.genes.resize(dim);

                if (uniformProb(rng) < cfg_.crossoverRate) {
                    double alpha = uniformProb(rng);
                    for (std::size_t d = 0; d < dim; ++d) {
                        c1.genes[d] = alpha * p1.genes[d] + (1.0 - alpha) * p2.genes[d];
                        c2.genes[d] = (1.0 - alpha) * p1.genes[d] + alpha * p2.genes[d];
                    }
                } else {
                    c1.genes = p1.genes;
                    c2.genes = p2.genes;
                }

                auto mutate = [&](Candidate& c) {
                    for (std::size_t d = 0; d < dim; ++d) {
                        if (uniformProb(rng) < cfg_.mutationRate) {
                            c.genes[d] += mutNoise(rng);
                        }
                    }
                    evaluateCandidate(c, iter);
                    evaluations++;
                };

                mutate(c1);
                nextPop.push_back(c1);
                if (nextPop.size() < popSize) {
                    mutate(c2);
                    nextPop.push_back(c2);
                }
            }

            pop = std::move(nextPop);
            updateBest();

            res.bestHistory.push_back(res.bestFitness);
            double sumFit = 0.0;
            for (const auto& c : pop) {
                sumFit += c.rawFitness;
            }
            res.avgHistory.push_back(sumFit / popSize);
        }

        res.evaluations = evaluations;
        res.generations = cfg_.search.iterations;
        return res;
    }

private:
    ConstrainedOptimizerConfig cfg_;
    ConstraintSet constraints_;
    ViolationEvaluator violationFn_;
    AdaptivePenaltyHandler penaltyHandler_;
};

} // namespace constraints
} // namespace ga
