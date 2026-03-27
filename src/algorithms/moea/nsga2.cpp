#include "ga/algorithms/moea/nsga2.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace ga {
namespace moea {

Nsga2::Nsga2(const Nsga2Config& cfg) : cfg_(cfg) {}

void Nsga2::validateObjectives(const std::vector<ga::Individual>& population) {
    if (population.empty()) {
        throw std::invalid_argument("Population cannot be empty");
    }

    const std::size_t objective_count = population.front().evaluation.objectives.size();
    if (objective_count == 0) {
        throw std::invalid_argument("Objective vectors must be non-empty");
    }

    for (const auto& individual : population) {
        if (individual.evaluation.objectives.size() != objective_count) {
            throw std::invalid_argument("All individuals must have the same objective dimension");
        }
    }
}

Nsga2GenerationStats Nsga2::summarizeGeneration(
    const std::vector<ga::Individual>& population,
    const std::vector<std::vector<std::size_t>>& fronts) {
    Nsga2GenerationStats stats;
    stats.firstFrontSize = fronts.empty() ? 0 : fronts.front().size();

    if (population.empty() || population.front().evaluation.objectives.empty()) {
        return stats;
    }

    double best_obj0 = std::numeric_limits<double>::infinity();
    for (const auto& ind : population) {
        best_obj0 = std::min(best_obj0, ind.evaluation.objectives[0]);
    }
    stats.bestObjective0 = best_obj0;
    return stats;
}

std::vector<std::size_t>
Nsga2::tournamentSelect(const std::vector<ga::Individual>& population,
                        std::size_t count,
                        std::mt19937& rng) const {
    validateObjectives(population);
    auto fronts = nonDominatedSort(population);

    std::vector<std::size_t> rank(population.size(), std::numeric_limits<std::size_t>::max());
    std::vector<double> crowding(population.size(), 0.0);

    for (std::size_t r = 0; r < fronts.size(); ++r) {
        for (std::size_t idx : fronts[r]) {
            rank[idx] = r;
        }
        auto distances = crowdingDistance(population, fronts[r]);
        for (std::size_t i = 0; i < fronts[r].size(); ++i) {
            crowding[fronts[r][i]] = distances[i];
        }
    }

    std::uniform_int_distribution<std::size_t> pick(0, population.size() - 1);
    std::vector<std::size_t> selected;
    selected.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t a = pick(rng);
        const std::size_t b = pick(rng);

        std::size_t winner = a;
        if (rank[b] < rank[a]) {
            winner = b;
        } else if (rank[b] == rank[a]) {
            if (crowding[b] > crowding[a]) {
                winner = b;
            } else if (crowding[b] == crowding[a]) {
                winner = (pick(rng) & 1U) ? a : b;
            }
        }
        selected.push_back(winner);
    }

    return selected;
}

std::vector<ga::Individual>
Nsga2::environmentalSelect(const std::vector<ga::Individual>& combined,
                           std::size_t targetSize) const {
    if (targetSize == 0) {
        return {};
    }
    validateObjectives(combined);

    auto fronts = nonDominatedSort(combined);
    std::vector<ga::Individual> next;
    next.reserve(targetSize);

    for (const auto& front : fronts) {
        if (front.empty()) {
            continue;
        }
        if (next.size() + front.size() <= targetSize) {
            for (std::size_t idx : front) {
                next.push_back(combined[idx]);
            }
            continue;
        }

        auto distances = crowdingDistance(combined, front);
        std::vector<std::size_t> order(front.size());
        for (std::size_t i = 0; i < front.size(); ++i) {
            order[i] = i;
        }

        std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
            const double dl = distances[lhs];
            const double dr = distances[rhs];
            if (dl == dr) {
                return front[lhs] < front[rhs];
            }
            return dl > dr;
        });

        const std::size_t remaining = targetSize - next.size();
        for (std::size_t i = 0; i < remaining; ++i) {
            next.push_back(combined[front[order[i]]]);
        }
        break;
    }

    return next;
}

Nsga2Result Nsga2::run(std::vector<ga::Individual> initialPopulation,
                       const EvaluateFn& evaluate,
                       const ReproduceFn& reproduce) const {
    if (!evaluate) {
        throw std::invalid_argument("NSGA-II evaluate callback must be provided");
    }
    if (!reproduce) {
        throw std::invalid_argument("NSGA-II reproduce callback must be provided");
    }
    if (initialPopulation.empty()) {
        throw std::invalid_argument("NSGA-II initial population cannot be empty");
    }

    const std::size_t target_size = cfg_.populationSize > 0
        ? cfg_.populationSize
        : initialPopulation.size();

    if (initialPopulation.size() != target_size) {
        throw std::invalid_argument("NSGA-II initial population size must match config.populationSize");
    }

    std::mt19937 rng(cfg_.seed == 0 ? std::random_device{}() : cfg_.seed);

    evaluate(initialPopulation);
    validateObjectives(initialPopulation);

    Nsga2Result result;
    auto current_fronts = nonDominatedSort(initialPopulation);
    result.history.push_back(summarizeGeneration(initialPopulation, current_fronts));

    auto population = std::move(initialPopulation);

    for (std::size_t gen = 0; gen < cfg_.generations; ++gen) {
        auto mating_pool = tournamentSelect(population, target_size, rng);

        auto offspring = reproduce(population, mating_pool, rng);
        if (offspring.empty()) {
            throw std::runtime_error("NSGA-II reproduction callback returned empty offspring");
        }

        if (offspring.size() > target_size) {
            offspring.resize(target_size);
        }
        while (offspring.size() < target_size) {
            const std::size_t idx = mating_pool[offspring.size() % mating_pool.size()];
            offspring.push_back(population[idx]);
        }

        evaluate(offspring);
        validateObjectives(offspring);

        std::vector<ga::Individual> combined;
        combined.reserve(population.size() + offspring.size());
        combined.insert(combined.end(), population.begin(), population.end());
        combined.insert(combined.end(), offspring.begin(), offspring.end());

        population = environmentalSelect(combined, target_size);
        current_fronts = nonDominatedSort(population);
        result.history.push_back(summarizeGeneration(population, current_fronts));
    }

    result.population = std::move(population);
    return result;
}

bool Nsga2::dominates(const ga::Individual& a, const ga::Individual& b) {
    if (a.evaluation.feasible != b.evaluation.feasible) {
        return a.evaluation.feasible && !b.evaluation.feasible;
    }

    const auto& ao = a.evaluation.objectives;
    const auto& bo = b.evaluation.objectives;

    if (ao.empty() || bo.empty() || ao.size() != bo.size()) {
        return false;
    }

    // Assume minimization objectives. Maximization can be represented by negating objectives.
    bool strictly_better_in_one = false;
    for (std::size_t i = 0; i < ao.size(); ++i) {
        if (ao[i] > bo[i]) {
            return false;
        }
        if (ao[i] < bo[i]) {
            strictly_better_in_one = true;
        }
    }
    return strictly_better_in_one;
}

std::vector<std::vector<std::size_t>>
Nsga2::nonDominatedSort(const std::vector<ga::Individual>& population) const {
    std::vector<std::vector<std::size_t>> fronts;
    if (population.empty()) {
        return fronts;
    }

    const std::size_t n = population.size();
    std::vector<std::vector<std::size_t>> dominates_list(n);
    std::vector<std::size_t> domination_count(n, 0);
    std::vector<std::size_t> first_front;

    for (std::size_t p = 0; p < n; ++p) {
        for (std::size_t q = 0; q < n; ++q) {
            if (p == q) {
                continue;
            }
            if (dominates(population[p], population[q])) {
                dominates_list[p].push_back(q);
            } else if (dominates(population[q], population[p])) {
                ++domination_count[p];
            }
        }
        if (domination_count[p] == 0) {
            first_front.push_back(p);
        }
    }

    fronts.push_back(first_front);
    std::size_t current = 0;

    while (current < fronts.size() && !fronts[current].empty()) {
        std::vector<std::size_t> next_front;
        for (std::size_t p : fronts[current]) {
            for (std::size_t q : dominates_list[p]) {
                if (domination_count[q] == 0) {
                    continue;
                }
                --domination_count[q];
                if (domination_count[q] == 0) {
                    next_front.push_back(q);
                }
            }
        }
        if (!next_front.empty()) {
            fronts.push_back(std::move(next_front));
        }
        ++current;
    }

    return fronts;
}

std::vector<double>
Nsga2::crowdingDistance(const std::vector<ga::Individual>& population,
                        const std::vector<std::size_t>& front) const {
    std::vector<double> distance(front.size(), 0.0);
    if (front.empty()) {
        return distance;
    }
    if (front.size() <= 2) {
        std::fill(distance.begin(), distance.end(), std::numeric_limits<double>::infinity());
        return distance;
    }

    const std::size_t objective_count = population[front[0]].evaluation.objectives.size();
    for (std::size_t idx : front) {
        if (population[idx].evaluation.objectives.size() != objective_count) {
            throw std::invalid_argument("Inconsistent objective vector size in front");
        }
    }

    std::vector<std::size_t> order(front.size());
    for (std::size_t i = 0; i < front.size(); ++i) {
        order[i] = i;
    }

    for (std::size_t m = 0; m < objective_count; ++m) {
        std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
            return population[front[lhs]].evaluation.objectives[m] <
                   population[front[rhs]].evaluation.objectives[m];
        });

        const double min_obj = population[front[order.front()]].evaluation.objectives[m];
        const double max_obj = population[front[order.back()]].evaluation.objectives[m];

        distance[order.front()] = std::numeric_limits<double>::infinity();
        distance[order.back()] = std::numeric_limits<double>::infinity();

        if (max_obj <= min_obj) {
            continue;
        }

        for (std::size_t k = 1; k + 1 < order.size(); ++k) {
            if (std::isinf(distance[order[k]])) {
                continue;
            }
            const double prev = population[front[order[k - 1]]].evaluation.objectives[m];
            const double next = population[front[order[k + 1]]].evaluation.objectives[m];
            distance[order[k]] += (next - prev) / (max_obj - min_obj);
        }
    }

    return distance;
}

} // namespace moea
} // namespace ga
