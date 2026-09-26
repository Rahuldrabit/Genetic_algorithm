#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <future>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ga/config.hpp"
#include "ga/core/result.hpp"
#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace algorithms {

enum class IslandTopology {
    Ring,
    Star,
    FullyConnected,
    RandomMesh
};

enum class MigrationPolicy {
    BestToWorst,
    RandomToWorst,
    BestToRandom
};

struct IslandConfig {
    std::size_t numIslands = 4;
    std::size_t islandPopulationSize = 50; // Total pop = numIslands * islandPopulationSize
    std::size_t iterations = 100;
    std::size_t migrationInterval = 10;
    std::size_t migrantsPerExchange = 2;
    IslandTopology topology = IslandTopology::Ring;
    MigrationPolicy policy = MigrationPolicy::BestToWorst;

    ga::metaheuristics::SearchConfig search;
    double crossoverRate = 0.8;
    double mutationRate = 0.1;
    double eliteRatio = 0.05;
};

class IslandModelOptimizer : public ga::metaheuristics::IContinuousOptimizer {
public:
    explicit IslandModelOptimizer(IslandConfig config = {})
        : cfg_(std::move(config)) {
        if (cfg_.numIslands == 0) cfg_.numIslands = 1;
        if (cfg_.islandPopulationSize == 0) cfg_.islandPopulationSize = 10;
        if (cfg_.migrationInterval == 0) cfg_.migrationInterval = 1;
        if (cfg_.migrantsPerExchange == 0) cfg_.migrantsPerExchange = 1;
    }

    std::string name() const override { return "IslandModelOptimizer"; }

    ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const ga::metaheuristics::SeedPopulation& seeds = {}) override {
        
        const std::size_t numIslands = cfg_.numIslands;
        const std::size_t islandPopSize = cfg_.islandPopulationSize;
        const std::size_t dim = cfg_.search.dimension > 0 ? cfg_.search.dimension : 10;
        const auto bounds = cfg_.search.bounds;
        const std::size_t totalIterations = cfg_.iterations > 0 ? cfg_.iterations : cfg_.search.iterations;

        struct Individual {
            std::vector<double> genes;
            double fitness = -std::numeric_limits<double>::infinity();
        };

        struct Island {
            std::vector<Individual> pop;
            std::mt19937 rng;
            Individual best;
        };

        std::vector<Island> islands(numIslands);
        std::mt19937 masterRng(cfg_.search.seed == 0 ? std::random_device{}() : cfg_.search.seed);

        std::size_t totalEvals = 0;
        std::uniform_real_distribution<double> geneDist(bounds.lower, bounds.upper);

        // Initialize islands
        for (std::size_t i = 0; i < numIslands; ++i) {
            islands[i].rng.seed(masterRng());
            islands[i].pop.resize(islandPopSize);
            islands[i].best.fitness = -std::numeric_limits<double>::infinity();

            for (std::size_t j = 0; j < islandPopSize; ++j) {
                islands[i].pop[j].genes.resize(dim);
                std::size_t seedIdx = i * islandPopSize + j;
                if (seedIdx < seeds.size() && seeds[seedIdx].size() == dim) {
                    islands[i].pop[j].genes = seeds[seedIdx];
                } else {
                    for (std::size_t d = 0; d < dim; ++d) {
                        islands[i].pop[j].genes[d] = geneDist(islands[i].rng);
                    }
                }
                islands[i].pop[j].fitness = fitness(islands[i].pop[j].genes);
                totalEvals++;

                if (islands[i].pop[j].fitness > islands[i].best.fitness) {
                    islands[i].best = islands[i].pop[j];
                }
            }
        }

        ga::core::OptimizationResult res;
        res.bestFitness = -std::numeric_limits<double>::infinity();
        res.bestSolution.resize(dim);

        // Track global best
        auto updateGlobalBest = [&]() {
            for (const auto& isl : islands) {
                if (isl.best.fitness > res.bestFitness) {
                    res.bestFitness = isl.best.fitness;
                    res.bestSolution = isl.best.genes;
                }
            }
        };
        updateGlobalBest();

        // Main generational loop with parallel island stepping
        for (std::size_t iter = 0; iter < totalIterations; ++iter) {
            std::vector<std::future<void>> futures;

            for (std::size_t i = 0; i < numIslands; ++i) {
                futures.push_back(std::async(std::launch::async, [&islands, i, dim, bounds, &fitness, this]() {
                    auto& island = islands[i];
                    auto& pop = island.pop;
                    const std::size_t popSize = pop.size();
                    std::uniform_real_distribution<double> prob(0.0, 1.0);
                    std::normal_distribution<double> norm(0.0, 0.1 * (bounds.upper - bounds.lower));

                    // Sort descending by fitness
                    std::sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
                        return a.fitness > b.fitness;
                    });

                    std::size_t eliteCount = std::max<std::size_t>(1, static_cast<std::size_t>(cfg_.eliteRatio * popSize));
                    std::vector<Individual> nextPop;
                    nextPop.reserve(popSize);

                    // Elitism
                    for (std::size_t e = 0; e < eliteCount; ++e) {
                        nextPop.push_back(pop[e]);
                    }

                    // Tournament selection helper
                    auto selectParent = [&](std::mt19937& rng) -> const Individual& {
                        std::uniform_int_distribution<std::size_t> dist(0, popSize - 1);
                        std::size_t i1 = dist(rng);
                        std::size_t i2 = dist(rng);
                        std::size_t i3 = dist(rng);
                        std::size_t bestIdx = i1;
                        if (pop[i2].fitness > pop[bestIdx].fitness) bestIdx = i2;
                        if (pop[i3].fitness > pop[bestIdx].fitness) bestIdx = i3;
                        return pop[bestIdx];
                    };

                    while (nextPop.size() < popSize) {
                        const auto& p1 = selectParent(island.rng);
                        const auto& p2 = selectParent(island.rng);

                        Individual c1, c2;
                        c1.genes.resize(dim);
                        c2.genes.resize(dim);

                        // Arithmetic crossover
                        if (prob(island.rng) < cfg_.crossoverRate) {
                            double alpha = prob(island.rng);
                            for (std::size_t d = 0; d < dim; ++d) {
                                c1.genes[d] = alpha * p1.genes[d] + (1.0 - alpha) * p2.genes[d];
                                c2.genes[d] = (1.0 - alpha) * p1.genes[d] + alpha * p2.genes[d];
                            }
                        } else {
                            c1.genes = p1.genes;
                            c2.genes = p2.genes;
                        }

                        // Gaussian mutation
                        auto mutate = [&](Individual& ind) {
                            for (std::size_t d = 0; d < dim; ++d) {
                                if (prob(island.rng) < cfg_.mutationRate) {
                                    ind.genes[d] += norm(island.rng);
                                    ind.genes[d] = std::clamp(ind.genes[d], bounds.lower, bounds.upper);
                                }
                            }
                            ind.fitness = fitness(ind.genes);
                        };

                        mutate(c1);
                        nextPop.push_back(c1);
                        if (nextPop.size() < popSize) {
                            mutate(c2);
                            nextPop.push_back(c2);
                        }
                    }

                    pop = std::move(nextPop);
                    for (const auto& ind : pop) {
                        if (ind.fitness > island.best.fitness) {
                            island.best = ind;
                        }
                    }
                }));
            }

            for (auto& f : futures) {
                f.get();
            }

            totalEvals += numIslands * (islandPopSize - std::max<std::size_t>(1, static_cast<std::size_t>(cfg_.eliteRatio * islandPopSize)));

            // Migration step
            if (numIslands > 1 && (iter + 1) % cfg_.migrationInterval == 0) {
                std::size_t k = std::min(cfg_.migrantsPerExchange, islandPopSize / 2);

                // Collect migrants from each island (top-k)
                std::vector<std::vector<Individual>> emigrants(numIslands);
                for (std::size_t i = 0; i < numIslands; ++i) {
                    std::sort(islands[i].pop.begin(), islands[i].pop.end(), [](const Individual& a, const Individual& b) {
                        return a.fitness > b.fitness;
                    });
                    for (std::size_t m = 0; m < k; ++m) {
                        emigrants[i].push_back(islands[i].pop[m]);
                    }
                }

                // Inject migrants based on topology
                for (std::size_t src = 0; src < numIslands; ++src) {
                    std::size_t dst = (src + 1) % numIslands; // Ring default
                    if (cfg_.topology == IslandTopology::Star) {
                        dst = (src == 0) ? (iter % (numIslands - 1) + 1) : 0;
                    }

                    auto& dstPop = islands[dst].pop;
                    std::sort(dstPop.begin(), dstPop.end(), [](const Individual& a, const Individual& b) {
                        return a.fitness > b.fitness;
                    });

                    // Replace worst individuals in destination
                    for (std::size_t m = 0; m < k && m < dstPop.size(); ++m) {
                        dstPop[dstPop.size() - 1 - m] = emigrants[src][m];
                    }
                }
            }

            updateGlobalBest();
            res.bestHistory.push_back(res.bestFitness);

            double sumFit = 0.0;
            for (const auto& isl : islands) {
                for (const auto& ind : isl.pop) {
                    sumFit += ind.fitness;
                }
            }
            res.avgHistory.push_back(sumFit / (numIslands * islandPopSize));
        }

        res.evaluations = totalEvals;
        res.generations = totalIterations;
        return res;
    }

private:
    IslandConfig cfg_;
};

} // namespace algorithms
} // namespace ga
