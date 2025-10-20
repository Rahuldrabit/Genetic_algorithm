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
    : cfg_(cfg), rng_(make_rng(cfg.seed)) {
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

std::vector<GeneticAlgorithm::Individual> GeneticAlgorithm::initPopulation_(const Fitness& f) {
    std::uniform_real_distribution<double> dist(cfg_.bounds.lower, cfg_.bounds.upper);
    std::vector<Individual> pop;
    pop.reserve(cfg_.populationSize);
    for (int i = 0; i < cfg_.populationSize; ++i) {
        Individual ind;
        ind.genes.resize(cfg_.dimension);
        for (double& g : ind.genes) g = dist(rng_);
        ind.fitness = f(ind.genes);
        pop.push_back(std::move(ind));
    }
    return pop;
}

pair<GeneticAlgorithm::Individual, GeneticAlgorithm::Individual>
GeneticAlgorithm::crossoverPair_(const Individual& p1, const Individual& p2, const Fitness& f) {
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    if (prob(rng_) < cfg_.crossoverRate) {
        auto children = crossover_->crossover(p1.genes, p2.genes);
        Individual c1{children.first, f(children.first)};
        Individual c2{children.second, f(children.second)};
        return {std::move(c1), std::move(c2)};
    }
    return {p1, p2};
}

void GeneticAlgorithm::mutate_(Individual& ind) {
    std::vector<double> lows(cfg_.dimension, cfg_.bounds.lower);
    std::vector<double> highs(cfg_.dimension, cfg_.bounds.upper);
    if (auto* g = dynamic_cast<GaussianMutation*>(mutation_.get())) {
        g->mutate(ind.genes, cfg_.mutationRate, 0.1, lows, highs);
    } else if (auto* u = dynamic_cast<UniformMutation*>(mutation_.get())) {
        u->mutate(ind.genes, cfg_.mutationRate, lows, highs);
    }
    for (double& x : ind.genes) {
        if (x < cfg_.bounds.lower) x = cfg_.bounds.lower;
        if (x > cfg_.bounds.upper) x = cfg_.bounds.upper;
    }
}

Result GeneticAlgorithm::run(const Fitness& fitness) {
    if (!crossover_ || !mutation_) throw std::runtime_error("Operators not set");

    auto pop = initPopulation_(fitness);

    Result res;
    res.bestHistory.reserve(cfg_.generations);
    res.avgHistory.reserve(cfg_.generations);

    std::uniform_int_distribution<int> pick(0, (int)pop.size() - 1);

    auto compute_stats = [&](const std::vector<Individual>& P){
        double sum = 0.0;
        double best = -1e300;
        size_t best_i = 0;
        for (size_t i = 0; i < P.size(); ++i) {
            sum += P[i].fitness;
            if (P[i].fitness > best) { best = P[i].fitness; best_i = i; }
        }
        res.bestGenes = P[best_i].genes;
        res.bestFitness = best;
        res.bestHistory.push_back(best);
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
            std::nth_element(idx.begin(), idx.begin()+elites, idx.end(), [&](size_t i, size_t j){ return pop[i].fitness > pop[j].fitness; });
            for (int i = 0; i < elites; ++i) next.push_back(pop[idx[i]]);
        }

        // Fill the rest
        while ((int)next.size() < cfg_.populationSize) {
            const auto& p1 = pop[pick(rng_)];
            const auto& p2 = pop[pick(rng_)];
            auto [c1, c2] = crossoverPair_(p1, p2, fitness);
            mutate_(c1);
            mutate_(c2);
            c1.fitness = fitness(c1.genes);
            c2.fitness = fitness(c2.genes);
            next.push_back(std::move(c1));
            if ((int)next.size() < cfg_.populationSize) next.push_back(std::move(c2));
        }

        pop.swap(next);
        compute_stats(pop);
    }

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
