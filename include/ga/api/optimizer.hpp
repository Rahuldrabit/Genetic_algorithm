#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ga/algorithms/moea/nsga2.hpp"
#include "ga/config.hpp"
#include "ga/evaluation/parallel_evaluator.hpp"
#include "ga/genetic_algorithm.hpp"
#include "ga/moea/nsga3.hpp"
#include "ga/representations/vector_genome.hpp"

namespace ga {
namespace api {

class Optimizer {
public:
    using Objective = std::function<double(const std::vector<double>&)>;

    struct MultiObjectiveResult {
        std::vector<std::vector<double>> paretoGenes;
        std::vector<std::vector<double>> paretoObjectives;
    };

    Optimizer& withConfig(ga::Config cfg) {
        cfg_ = std::move(cfg);
        return *this;
    }

    Optimizer& withThreads(std::size_t threads) {
        threads_ = threads;
        return *this;
    }

    Optimizer& withSeed(unsigned seed) {
        cfg_.seed = seed;
        return *this;
    }

    ga::Result optimize(const Objective& objective) const {
        if (!objective) {
            throw std::invalid_argument("Optimizer objective callback is empty");
        }
        ga::GeneticAlgorithm ga(cfg_);
        return ga.run(objective);
    }

    MultiObjectiveResult optimizeMultiObjective(const std::vector<Objective>& objectives,
                                                std::size_t populationSize = 80,
                                                std::size_t generations = 80) const {
        return optimizeMultiObjectiveNsga2(objectives, populationSize, generations);
    }

    MultiObjectiveResult optimizeMultiObjectiveNsga2(const std::vector<Objective>& objectives,
                                                     std::size_t populationSize = 80,
                                                     std::size_t generations = 80) const {
        if (objectives.empty()) {
            throw std::invalid_argument("At least one objective is required");
        }

        ga::moea::Nsga2Config ncfg;
        ncfg.populationSize = populationSize;
        ncfg.generations = generations;
        ncfg.seed = cfg_.seed;

        ga::moea::Nsga2 nsga2(ncfg);

        std::mt19937 rng(cfg_.seed == 0 ? std::random_device{}() : cfg_.seed);
        std::uniform_real_distribution<double> init(cfg_.bounds.lower, cfg_.bounds.upper);

        std::vector<ga::Individual> initial;
        initial.reserve(populationSize);
        for (std::size_t i = 0; i < populationSize; ++i) {
            std::vector<double> genes(static_cast<std::size_t>(cfg_.dimension));
            for (double& g : genes) {
                g = init(rng);
            }
            ga::Individual ind;
            ind.genome = std::make_unique<ga::representations::VectorGenome<double>>(std::move(genes));
            initial.push_back(std::move(ind));
        }

        auto evaluate = [&](std::vector<ga::Individual>& pop) {
            evaluatePopulation(pop, objectives, threads_);
        };

        auto reproduce = [&](const std::vector<ga::Individual>& parents,
                             const std::vector<std::size_t>& mating,
                             std::mt19937& rrng) {
            std::normal_distribution<double> noise(0.0, 0.1);
            std::vector<ga::Individual> offspring;
            offspring.reserve(mating.size());
            for (std::size_t idx : mating) {
                auto* pg = dynamic_cast<const ga::representations::VectorGenome<double>*>(parents[idx].genome.get());
                if (!pg) {
                    throw std::runtime_error("Optimizer expected VectorGenome<double> parent");
                }
                std::vector<double> child = pg->genes;
                for (double& g : child) {
                    g = std::clamp(g + noise(rrng), cfg_.bounds.lower, cfg_.bounds.upper);
                }
                ga::Individual ind;
                ind.genome = std::make_unique<ga::representations::VectorGenome<double>>(std::move(child));
                offspring.push_back(std::move(ind));
            }
            return offspring;
        };

        const auto run = nsga2.run(std::move(initial), evaluate, reproduce);
        const auto fronts = nsga2.nonDominatedSort(run.population);

        return collectFirstFront(run.population, fronts);
    }

    MultiObjectiveResult optimizeMultiObjectiveNsga3(const std::vector<Objective>& objectives,
                                                     std::size_t populationSize = 80,
                                                     std::size_t generations = 80,
                                                     std::size_t referenceDivisions = 8) const {
        if (objectives.empty()) {
            throw std::invalid_argument("At least one objective is required");
        }

        ga::moea::Nsga3 nsga3({populationSize, generations, cfg_.seed});
        auto population = makeInitialPopulation(populationSize);
        evaluatePopulation(population, objectives, threads_);

        const auto referencePoints =
            ga::moea::Nsga3::generateDasDennisReferencePoints(objectives.size(), referenceDivisions);

        auto reproduce = [&](const std::vector<ga::Individual>& parents,
                             const std::vector<std::size_t>& mating,
                             std::mt19937& rrng) {
            std::normal_distribution<double> noise(0.0, 0.1);
            std::vector<ga::Individual> offspring;
            offspring.reserve(mating.size());
            for (std::size_t idx : mating) {
                auto* pg =
                    dynamic_cast<const ga::representations::VectorGenome<double>*>(parents[idx].genome.get());
                if (!pg) {
                    throw std::runtime_error("Optimizer expected VectorGenome<double> parent");
                }
                std::vector<double> child = pg->genes;
                for (double& g : child) {
                    g = std::clamp(g + noise(rrng), cfg_.bounds.lower, cfg_.bounds.upper);
                }
                ga::Individual ind;
                ind.genome = std::make_unique<ga::representations::VectorGenome<double>>(std::move(child));
                offspring.push_back(std::move(ind));
            }
            return offspring;
        };

        std::mt19937 rng(cfg_.seed == 0 ? std::random_device{}() : cfg_.seed + 31u);
        for (std::size_t gen = 0; gen < generations; ++gen) {
            std::uniform_int_distribution<std::size_t> pick(0, population.size() - 1);
            std::vector<std::size_t> mating(populationSize);
            for (std::size_t& i : mating) {
                i = pick(rng);
            }

            auto offspring = reproduce(population, mating, rng);
            evaluatePopulation(offspring, objectives, threads_);

            std::vector<ga::Individual> combined;
            combined.reserve(population.size() + offspring.size());
            combined.insert(combined.end(), population.begin(), population.end());
            combined.insert(combined.end(), offspring.begin(), offspring.end());

            population = nsga3.environmentalSelect(combined, populationSize, referencePoints);
        }

        const auto fronts = nsga3.nonDominatedSort(population);
        return collectFirstFront(population, fronts);
    }

private:
    static MultiObjectiveResult
    collectFirstFront(const std::vector<ga::Individual>& population,
                      const std::vector<std::vector<std::size_t>>& fronts) {
        MultiObjectiveResult out;
        if (fronts.empty()) {
            return out;
        }

        for (std::size_t idx : fronts.front()) {
            const auto* vg = dynamic_cast<const ga::representations::VectorGenome<double>*>(
                population[idx].genome.get());
            if (!vg) {
                continue;
            }
            out.paretoGenes.push_back(vg->genes);
            out.paretoObjectives.push_back(population[idx].evaluation.objectives);
        }
        return out;
    }

    std::vector<ga::Individual> makeInitialPopulation(std::size_t populationSize) const {
        std::mt19937 rng(cfg_.seed == 0 ? std::random_device{}() : cfg_.seed);
        std::uniform_real_distribution<double> init(cfg_.bounds.lower, cfg_.bounds.upper);

        std::vector<ga::Individual> initial;
        initial.reserve(populationSize);
        for (std::size_t i = 0; i < populationSize; ++i) {
            std::vector<double> genes(static_cast<std::size_t>(cfg_.dimension));
            for (double& g : genes) {
                g = init(rng);
            }
            ga::Individual ind;
            ind.genome = std::make_unique<ga::representations::VectorGenome<double>>(std::move(genes));
            initial.push_back(std::move(ind));
        }
        return initial;
    }

    static void evaluatePopulation(std::vector<ga::Individual>& pop,
                                   const std::vector<Objective>& objectives,
                                   std::size_t threads) {
        std::vector<std::vector<double>> genes;
        genes.reserve(pop.size());
        for (const auto& ind : pop) {
            const auto* vg = dynamic_cast<const ga::representations::VectorGenome<double>*>(ind.genome.get());
            if (!vg) {
                throw std::runtime_error("Optimizer expected VectorGenome<double>");
            }
            genes.push_back(vg->genes);
        }

        auto evalOne = [&](const std::vector<double>& g) {
            std::vector<double> vals;
            vals.reserve(objectives.size());
            for (const auto& obj : objectives) {
                vals.push_back(obj(g));
            }
            return vals;
        };

        std::vector<std::vector<double>> values;
        if (threads > 1) {
            ga::evaluation::ParallelEvaluator<std::vector<double>, std::vector<double>, decltype(evalOne)> pe(
                evalOne, threads);
            values = pe.evaluate(genes);
        } else {
            values.reserve(genes.size());
            for (const auto& g : genes) {
                values.push_back(evalOne(g));
            }
        }

        for (std::size_t i = 0; i < pop.size(); ++i) {
            pop[i].evaluation.objectives = std::move(values[i]);
            pop[i].evaluation.feasible = true;
            pop[i].evaluation.penalty = 0.0;
        }
    }

    ga::Config cfg_{};
    std::size_t threads_ = 1;
};

} // namespace api
} // namespace ga
