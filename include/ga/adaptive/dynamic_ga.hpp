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

#include "ga/adaptive/adaptive_policy.hpp"
#include "ga/config.hpp"
#include "ga/core/result.hpp"
#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace adaptive {

enum class DynamicStrategy {
    Hypermutation,
    RandomImmigrants,
    MemoryArchive,
    Hybrid
};

struct DynamicGAConfig {
    ga::metaheuristics::SearchConfig search;
    std::size_t stagnationWindow = 8;
    double stagnationTolerance = 1e-6;
    DynamicStrategy strategy = DynamicStrategy::Hybrid;
    double immigrantRatio = 0.25;          // Fraction of worst population replaced
    double hypermutationFactor = 5.0;      // Mutation rate multiplier during hypermutation
    std::size_t hypermutationDuration = 4; // Iterations hypermutation stays active
    double diversityThreshold = 0.08;      // Below this diversity, diversity injection triggers
    bool detectEnvironmentChange = true;   // Re-evaluates best to detect landscape shifts
    std::size_t memorySize = 10;           // Archive size for dynamic memory
    double baseCrossoverRate = 0.8;
    double baseMutationRate = 0.05;
};

class DynamicGAOptimizer : public ga::metaheuristics::IContinuousOptimizer {
public:
    explicit DynamicGAOptimizer(DynamicGAConfig config = {})
        : cfg_(std::move(config)) {
        if (cfg_.stagnationWindow == 0) cfg_.stagnationWindow = 5;
        if (cfg_.immigrantRatio < 0.0 || cfg_.immigrantRatio > 0.8) cfg_.immigrantRatio = 0.25;
        if (cfg_.hypermutationFactor <= 1.0) cfg_.hypermutationFactor = 5.0;
        if (cfg_.memorySize == 0) cfg_.memorySize = 5;
    }

    std::string name() const override { return "DynamicGAOptimizer"; }

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

        struct Ind {
            std::vector<double> genes;
            double fitness = -std::numeric_limits<double>::infinity();
        };

        std::vector<Ind> pop(popSize);
        std::vector<Ind> memoryArchive;
        memoryArchive.reserve(cfg_.memorySize);

        std::size_t evaluations = 0;

        // Initialize population
        for (std::size_t i = 0; i < popSize; ++i) {
            pop[i].genes.resize(dim);
            if (i < seeds.size() && seeds[i].size() == dim) {
                pop[i].genes = seeds[i];
                ga::metaheuristics::detail::clampToBounds(pop[i].genes, bounds);
            } else {
                for (std::size_t d = 0; d < dim; ++d) {
                    pop[i].genes[d] = initDist(rng);
                }
            }
            pop[i].fitness = fitness(pop[i].genes);
            evaluations++;
        }

        ga::core::OptimizationResult res;
        res.bestFitness = -std::numeric_limits<double>::infinity();
        res.bestSolution.resize(dim);
        res.bestHistory.reserve(cfg_.search.iterations);
        res.avgHistory.reserve(cfg_.search.iterations);

        std::size_t stagnationCounter = 0;
        std::size_t hypermutationCounter = 0;
        double prevBestFitness = -std::numeric_limits<double>::infinity();

        auto updateBest = [&]() {
            for (const auto& ind : pop) {
                if (ind.fitness > res.bestFitness) {
                    res.bestFitness = ind.fitness;
                    res.bestSolution = ind.genes;

                    // Update memory archive
                    if (memoryArchive.size() < cfg_.memorySize) {
                        memoryArchive.push_back(ind);
                    } else {
                        // Replace lowest in memory
                        auto minMem = std::min_element(memoryArchive.begin(), memoryArchive.end(),
                            [](const Ind& a, const Ind& b) { return a.fitness < b.fitness; });
                        if (ind.fitness > minMem->fitness) {
                            *minMem = ind;
                        }
                    }
                }
            }
        };

        updateBest();
        prevBestFitness = res.bestFitness;

        for (std::size_t iter = 0; iter < cfg_.search.iterations; ++iter) {
            // Check for environmental shift: re-evaluate the best solution
            if (cfg_.detectEnvironmentChange && iter > 0) {
                double reEval = fitness(res.bestSolution);
                evaluations++;
                if (std::abs(reEval - res.bestFitness) > 1e-4) {
                    // Landscape shifted! Re-evaluate all individuals and trigger adaptation
                    for (auto& ind : pop) {
                        ind.fitness = fitness(ind.genes);
                        evaluations++;
                    }
                    res.bestFitness = -std::numeric_limits<double>::infinity();
                    updateBest();
                    stagnationCounter = cfg_.stagnationWindow; // force adaptation
                }
            }

            // Compute population diversity
            double avgVariance = 0.0;
            std::vector<double> mean(dim, 0.0);
            for (const auto& ind : pop) {
                for (std::size_t d = 0; d < dim; ++d) {
                    mean[d] += ind.genes[d] / popSize;
                }
            }
            for (const auto& ind : pop) {
                for (std::size_t d = 0; d < dim; ++d) {
                    double diff = ind.genes[d] - mean[d];
                    avgVariance += (diff * diff) / (dim * popSize);
                }
            }
            double normalizedDiversity = std::sqrt(avgVariance) / (boundSpan + 1e-9);

            // Stagnation detection
            if (res.bestFitness - prevBestFitness < cfg_.stagnationTolerance) {
                stagnationCounter++;
            } else {
                stagnationCounter = 0;
                prevBestFitness = res.bestFitness;
            }

            bool triggerAdaptation = (stagnationCounter >= cfg_.stagnationWindow) ||
                                     (normalizedDiversity < cfg_.diversityThreshold);

            if (triggerAdaptation && hypermutationCounter == 0) {
                hypermutationCounter = cfg_.hypermutationDuration;
                stagnationCounter = 0;
            }

            // Determine effective mutation rate
            double effectiveMutationRate = cfg_.baseMutationRate;
            if (hypermutationCounter > 0) {
                effectiveMutationRate = std::min(0.8, cfg_.baseMutationRate * cfg_.hypermutationFactor);
                hypermutationCounter--;
            }

            std::normal_distribution<double> mutNoise(0.0, 0.1 * boundSpan);

            // Sort population
            std::sort(pop.begin(), pop.end(), [](const Ind& a, const Ind& b) {
                return a.fitness > b.fitness;
            });

            // Random Immigrants: replace bottom fraction with random or memory archive
            if (triggerAdaptation && (cfg_.strategy == DynamicStrategy::RandomImmigrants || cfg_.strategy == DynamicStrategy::Hybrid)) {
                std::size_t numImmigrants = static_cast<std::size_t>(cfg_.immigrantRatio * popSize);
                for (std::size_t i = 0; i < numImmigrants; ++i) {
                    std::size_t replaceIdx = popSize - 1 - i;
                    if (!memoryArchive.empty() && uniformProb(rng) < 0.3) {
                        // Re-introduce memory migrant with slight perturbation
                        std::uniform_int_distribution<std::size_t> memDist(0, memoryArchive.size() - 1);
                        pop[replaceIdx] = memoryArchive[memDist(rng)];
                        for (std::size_t d = 0; d < dim; ++d) {
                            pop[replaceIdx].genes[d] = std::clamp(pop[replaceIdx].genes[d] + mutNoise(rng), bounds.lower, bounds.upper);
                        }
                    } else {
                        // Fresh random immigrant
                        for (std::size_t d = 0; d < dim; ++d) {
                            pop[replaceIdx].genes[d] = initDist(rng);
                        }
                    }
                    pop[replaceIdx].fitness = fitness(pop[replaceIdx].genes);
                    evaluations++;
                }

                // Re-sort after immigrant insertion
                std::sort(pop.begin(), pop.end(), [](const Ind& a, const Ind& b) {
                    return a.fitness > b.fitness;
                });
            }

            // Generate next generation
            std::vector<Ind> nextPop;
            nextPop.reserve(popSize);

            // Elite preservation
            nextPop.push_back(pop[0]);

            auto tournamentSelect = [&](std::mt19937& r) -> const Ind& {
                std::uniform_int_distribution<std::size_t> dist(0, popSize - 1);
                std::size_t a = dist(r);
                std::size_t b = dist(r);
                return pop[a].fitness > pop[b].fitness ? pop[a] : pop[b];
            };

            while (nextPop.size() < popSize) {
                const auto& p1 = tournamentSelect(rng);
                const auto& p2 = tournamentSelect(rng);

                Ind c1, c2;
                c1.genes.resize(dim);
                c2.genes.resize(dim);

                if (uniformProb(rng) < cfg_.baseCrossoverRate) {
                    double alpha = uniformProb(rng);
                    for (std::size_t d = 0; d < dim; ++d) {
                        c1.genes[d] = alpha * p1.genes[d] + (1.0 - alpha) * p2.genes[d];
                        c2.genes[d] = (1.0 - alpha) * p1.genes[d] + alpha * p2.genes[d];
                    }
                } else {
                    c1.genes = p1.genes;
                    c2.genes = p2.genes;
                }

                auto applyMutation = [&](Ind& ind) {
                    for (std::size_t d = 0; d < dim; ++d) {
                        if (uniformProb(rng) < effectiveMutationRate) {
                            ind.genes[d] = std::clamp(ind.genes[d] + mutNoise(rng), bounds.lower, bounds.upper);
                        }
                    }
                    ind.fitness = fitness(ind.genes);
                    evaluations++;
                };

                applyMutation(c1);
                nextPop.push_back(c1);
                if (nextPop.size() < popSize) {
                    applyMutation(c2);
                    nextPop.push_back(c2);
                }
            }

            pop = std::move(nextPop);
            updateBest();

            res.bestHistory.push_back(res.bestFitness);
            double sumFit = 0.0;
            for (const auto& ind : pop) {
                sumFit += ind.fitness;
            }
            res.avgHistory.push_back(sumFit / popSize);
        }

        res.evaluations = evaluations;
        res.generations = cfg_.search.iterations;
        return res;
    }

private:
    DynamicGAConfig cfg_;
};

} // namespace adaptive
} // namespace ga
