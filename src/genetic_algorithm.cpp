#include "ga/genetic_algorithm.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// Use existing operators
#include "mutation/gaussian_mutation.h"
#include "mutation/uniform_mutation.h"
#include "crossover/one_point_crossover.h"
#include "crossover/two_point_crossover.h"

using namespace std;

namespace ga {

static std::mt19937 make_rng(unsigned seed) {
    if (seed == 0) {
        std::random_device rd;
        return std::mt19937{rd()};
    }
    return std::mt19937{seed};
}

GeneticAlgorithm::GeneticAlgorithm(const Config& cfg)
    : cfg_(cfg),
      rng_(make_rng(cfg.seed)),
      lowerBounds_(static_cast<std::size_t>(std::max(0, cfg.dimension)), cfg.bounds.lower),
      upperBounds_(static_cast<std::size_t>(std::max(0, cfg.dimension)), cfg.bounds.upper) {
    if (cfg_.populationSize <= 0 || cfg_.generations < 0 || cfg_.dimension <= 0 ||
        !std::isfinite(cfg_.bounds.lower) || !std::isfinite(cfg_.bounds.upper) ||
        !std::isfinite(cfg_.bounds.upper - cfg_.bounds.lower) ||
        !std::isfinite(cfg_.crossoverRate) || !std::isfinite(cfg_.mutationRate) ||
        !std::isfinite(cfg_.eliteRatio) ||
        cfg_.bounds.lower >= cfg_.bounds.upper || cfg_.crossoverRate < 0.0 ||
        cfg_.crossoverRate > 1.0 || cfg_.mutationRate < 0.0 ||
        cfg_.mutationRate > 1.0 || cfg_.eliteRatio < 0.0 || cfg_.eliteRatio > 1.0) {
        throw std::invalid_argument("Invalid genetic algorithm configuration");
    }
    // Default operators
    mutation_ = makeGaussianMutation(cfg.seed);
    crossover_ = makeOnePointCrossover(cfg.seed);
}

GeneticAlgorithm::~GeneticAlgorithm() = default;

void GeneticAlgorithm::setMutationOperator(std::unique_ptr<MutationOperator> op) {
    mutation_ = std::move(op);
}

void GeneticAlgorithm::setCrossoverOperator(std::unique_ptr<CrossoverOperator> op) {
    crossover_ = std::move(op);
}

std::vector<GeneticAlgorithm::Individual> GeneticAlgorithm::initPopulation_(
    const Fitness& f,
    const std::vector<std::vector<double>>& initialSolutions,
    std::size_t& evaluations) {
    std::uniform_real_distribution<double> dist(cfg_.bounds.lower, cfg_.bounds.upper);
    std::vector<Individual> pop;
    pop.reserve(cfg_.populationSize);

    for (const auto& seed : initialSolutions) {
        if (static_cast<int>(pop.size()) == cfg_.populationSize) {
            break;
        }
        if (seed.size() != static_cast<std::size_t>(cfg_.dimension)) {
            throw std::invalid_argument("Initial solution dimension does not match Config");
        }
        if (!std::all_of(seed.begin(), seed.end(), [](double value) {
                return std::isfinite(value);
            })) {
            throw std::invalid_argument("Initial solutions must contain only finite values");
        }
        Individual ind;
        ind.genes = seed;
        for (double& gene : ind.genes) {
            gene = std::clamp(gene, cfg_.bounds.lower, cfg_.bounds.upper);
        }
        ind.fitness = f(ind.genes);
        if (!std::isfinite(ind.fitness)) {
            throw std::domain_error("Fitness callback returned a non-finite value");
        }
        ++evaluations;
        pop.push_back(std::move(ind));
    }

    while (static_cast<int>(pop.size()) < cfg_.populationSize) {
        Individual ind;
        ind.genes.resize(cfg_.dimension);
        for (double& g : ind.genes) g = dist(rng_);
        ind.fitness = f(ind.genes);
        if (!std::isfinite(ind.fitness)) {
            throw std::domain_error("Fitness callback returned a non-finite value");
        }
        ++evaluations;
        pop.push_back(std::move(ind));
    }
    return pop;
}

pair<GeneticAlgorithm::Individual, GeneticAlgorithm::Individual>
GeneticAlgorithm::crossoverPair_(const Individual& p1, const Individual& p2) {
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    if (prob(rng_) < cfg_.crossoverRate) {
        auto children = crossover_->crossover(p1.genes, p2.genes);
        Individual c1{std::move(children.first), 0.0};
        Individual c2{std::move(children.second), 0.0};
        return {std::move(c1), std::move(c2)};
    }
    return {p1, p2};
}

void GeneticAlgorithm::mutate_(Individual& ind) {
    if (auto* g = dynamic_cast<GaussianMutation*>(mutation_.get())) {
        g->mutate(ind.genes, cfg_.mutationRate, 0.1, lowerBounds_, upperBounds_);
    } else if (auto* u = dynamic_cast<UniformMutation*>(mutation_.get())) {
        u->mutate(ind.genes, cfg_.mutationRate, lowerBounds_, upperBounds_);
    }
    for (double& x : ind.genes) {
        if (x < cfg_.bounds.lower) x = cfg_.bounds.lower;
        if (x > cfg_.bounds.upper) x = cfg_.bounds.upper;
    }
}

Result GeneticAlgorithm::run(const Fitness& fitness) {
    return run(fitness, {});
}

Result GeneticAlgorithm::run(
    const Fitness& fitness,
    const std::vector<std::vector<double>>& initialSolutions) {
    if (!crossover_ || !mutation_) throw std::runtime_error("Operators not set");
    if (!fitness) throw std::invalid_argument("Fitness callback is empty");

    std::size_t evaluations = 0;
    auto pop = initPopulation_(fitness, initialSolutions, evaluations);

    Result res;
    res.bestHistory.reserve(static_cast<std::size_t>(cfg_.generations) + 1);
    res.avgHistory.reserve(static_cast<std::size_t>(cfg_.generations) + 1);

    std::uniform_int_distribution<int> pick(0, (int)pop.size() - 1);

    auto compute_stats = [&](const std::vector<Individual>& P){
        double sum = 0.0;
        double best = -1e300;
        size_t best_i = 0;
        for (size_t i = 0; i < P.size(); ++i) {
            sum += P[i].fitness;
            if (P[i].fitness > best) { best = P[i].fitness; best_i = i; }
        }
        if (res.bestGenes.empty() || best > res.bestFitness) {
            res.bestGenes = P[best_i].genes;
            res.bestFitness = best;
        }
        res.bestHistory.push_back(res.bestFitness);
        res.avgHistory.push_back(sum / P.size());
    };

    compute_stats(pop);

    for (int gen = 0; gen < cfg_.generations; ++gen) {
        // Elitism count
        int elites = std::max(0, (int)std::round(cfg_.eliteRatio * pop.size()));
        // Keep top elites
        std::vector<Individual> next;
        next.reserve(pop.size());
        if (elites > 0) {
            std::vector<size_t> idx(pop.size());
            std::iota(idx.begin(), idx.end(), 0);
            if (elites < static_cast<int>(idx.size())) {
                std::nth_element(idx.begin(), idx.begin()+elites, idx.end(), [&](size_t i, size_t j){ return pop[i].fitness > pop[j].fitness; });
            }
            for (int i = 0; i < elites; ++i) next.push_back(pop[idx[i]]);
        }

        // Fill the rest
        while ((int)next.size() < cfg_.populationSize) {
            const auto& p1 = pop[pick(rng_)];
            const auto& p2 = pop[pick(rng_)];
            auto [c1, c2] = crossoverPair_(p1, p2);
            mutate_(c1);
            c1.fitness = fitness(c1.genes);
            if (!std::isfinite(c1.fitness)) {
                throw std::domain_error("Fitness callback returned a non-finite value");
            }
            ++evaluations;
            next.push_back(std::move(c1));
            if ((int)next.size() < cfg_.populationSize) {
                mutate_(c2);
                c2.fitness = fitness(c2.genes);
                if (!std::isfinite(c2.fitness)) {
                    throw std::domain_error("Fitness callback returned a non-finite value");
                }
                ++evaluations;
                next.push_back(std::move(c2));
            }
        }

        pop.swap(next);
        compute_stats(pop);
        res.iterations = static_cast<std::size_t>(gen + 1);
    }

    res.evaluations = evaluations;
    return res;
}

// Factories
std::unique_ptr<MutationOperator> makeGaussianMutation(unsigned seed) {
    if (seed == 0) return std::make_unique<GaussianMutation>();
    return std::make_unique<GaussianMutation>(seed);
}
std::unique_ptr<MutationOperator> makeUniformMutation(unsigned seed) {
    if (seed == 0) return std::make_unique<UniformMutation>();
    return std::make_unique<UniformMutation>(seed);
}
std::unique_ptr<CrossoverOperator> makeOnePointCrossover(unsigned seed) {
    if (seed == 0) return std::make_unique<OnePointCrossover>();
    return std::make_unique<OnePointCrossover>(seed);
}
std::unique_ptr<CrossoverOperator> makeTwoPointCrossover(unsigned seed) {
    if (seed == 0) return std::make_unique<TwoPointCrossover>();
    return std::make_unique<TwoPointCrossover>(seed);
}

} // namespace ga
