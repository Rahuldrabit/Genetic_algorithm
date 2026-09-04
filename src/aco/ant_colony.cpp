#include "ga/aco/ant_colony.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace ga {
namespace aco {
namespace {

struct Tour {
    std::vector<std::size_t> nodes;
    double cost = std::numeric_limits<double>::infinity();
};

void validate(const AntColonyConfig& config) {
    const bool finite = std::isfinite(config.alpha) && std::isfinite(config.beta) &&
                        std::isfinite(config.evaporation) &&
                        std::isfinite(config.depositScale) &&
                        std::isfinite(config.initialPheromone) &&
                        std::isfinite(config.elitistWeight) &&
                        std::isfinite(config.exploitationProbability) &&
                        std::isfinite(config.localEvaporation) &&
                        std::isfinite(config.minPheromone) &&
                        std::isfinite(config.maxPheromone);
    if (!finite || config.ants == 0 || config.iterations == 0 || config.alpha < 0.0 ||
        config.beta < 0.0 || config.evaporation <= 0.0 ||
        config.evaporation >= 1.0 || config.depositScale <= 0.0 ||
        config.initialPheromone <= 0.0 || config.elitistWeight < 0.0 ||
        config.rankCount == 0 || config.exploitationProbability < 0.0 ||
        config.exploitationProbability > 1.0 || config.localEvaporation <= 0.0 ||
        config.localEvaporation >= 1.0 || config.minPheromone < 0.0 ||
        config.maxPheromone < 0.0 ||
        (config.maxPheromone > 0.0 &&
         config.minPheromone > config.maxPheromone)) {
        throw std::invalid_argument("invalid ant-colony configuration");
    }
}

void updateEdge(std::vector<double>& pheromone,
                std::size_t size,
                bool symmetric,
                std::size_t from,
                std::size_t to,
                double delta) {
    pheromone[from * size + to] += delta;
    if (symmetric) {
        pheromone[to * size + from] += delta;
    }
}

void depositTour(std::vector<double>& pheromone,
                 std::size_t size,
                 bool symmetric,
                 const Tour& tour,
                 double amount) {
    for (std::size_t i = 0; i < size; ++i) {
        updateEdge(pheromone,
                   size,
                   symmetric,
                   tour.nodes[i],
                   tour.nodes[(i + 1) % size],
                   amount);
    }
}

double routeDiversity(const std::vector<Tour>& tours, std::size_t size) {
    if (tours.size() < 2 || size < 2) {
        return 0.0;
    }
    std::unordered_set<std::size_t> edges;
    edges.reserve(tours.size() * size);
    for (const auto& tour : tours) {
        for (std::size_t i = 0; i < size; ++i) {
            const std::size_t from = tour.nodes[i];
            const std::size_t to = tour.nodes[(i + 1) % size];
            edges.insert(from * size + to);
        }
    }
    const double minimumEdges = static_cast<double>(size);
    const double maximumEdges = static_cast<double>(std::min(size * size, tours.size() * size));
    if (maximumEdges <= minimumEdges) {
        return 0.0;
    }
    return std::clamp((static_cast<double>(edges.size()) - minimumEdges) /
                          (maximumEdges - minimumEdges),
                      0.0,
                      1.0);
}

} // namespace

DenseGraph::DenseGraph(std::vector<std::vector<double>> costs, bool symmetric)
    : size_(costs.size()), symmetric_(symmetric) {
    if (size_ < 2) {
        throw std::invalid_argument("ACO graph must contain at least two nodes");
    }
    costs_.reserve(size_ * size_);
    for (std::size_t i = 0; i < size_; ++i) {
        if (costs[i].size() != size_) {
            throw std::invalid_argument("ACO cost matrix must be square");
        }
        for (std::size_t j = 0; j < size_; ++j) {
            const double value = costs[i][j];
            if (!std::isfinite(value) || (i != j && value <= 0.0) ||
                (i == j && value < 0.0)) {
                throw std::invalid_argument(
                    "ACO costs must be finite, with positive off-diagonal entries");
            }
            costs_.push_back(value);
        }
    }
    if (symmetric_) {
        for (std::size_t i = 0; i < size_; ++i) {
            for (std::size_t j = i + 1; j < size_; ++j) {
                const double scale = std::max({1.0, std::abs(cost(i, j)), std::abs(cost(j, i))});
                if (std::abs(cost(i, j) - cost(j, i)) > 1e-12 * scale) {
                    throw std::invalid_argument(
                        "symmetric ACO graph requires a symmetric cost matrix");
                }
            }
        }
    }
}

AntColonyOptimizer::AntColonyOptimizer(AntColonyConfig config)
    : config_(std::move(config)) {
    validate(config_);
}

AntColonyResult AntColonyOptimizer::solve(const DenseGraph& graph) const {
    const std::size_t size = graph.size();
    std::mt19937 rng = ga::metaheuristics::detail::makeRng(config_.seed);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> startNode(0, size - 1);

    std::vector<double> pheromone(size * size, config_.initialPheromone);
    std::vector<double> heuristic(size * size, 0.0);
    std::vector<std::vector<std::size_t>> candidateLists(size);
    for (std::size_t from = 0; from < size; ++from) {
        candidateLists[from].reserve(size - 1);
        for (std::size_t to = 0; to < size; ++to) {
            if (from != to) {
                heuristic[from * size + to] = 1.0 / graph.cost(from, to);
                candidateLists[from].push_back(to);
            }
        }
        std::sort(candidateLists[from].begin(), candidateLists[from].end(),
                  [&](std::size_t a, std::size_t b) {
                      return graph.cost(from, a) < graph.cost(from, b);
                  });
        if (config_.candidateListSize > 0 &&
            candidateLists[from].size() > config_.candidateListSize) {
            candidateLists[from].resize(config_.candidateListSize);
        }
    }

    AntColonyResult result;
    result.bestCost = std::numeric_limits<double>::infinity();
    result.bestCostHistory.reserve(config_.iterations);
    std::size_t stagnant = 0;
    double previousImprovement = 1.0;
    ga::metaheuristics::ControlSignal control;

    for (std::size_t iteration = 0; iteration < config_.iterations; ++iteration) {
        std::vector<Tour> tours(config_.ants);
        for (Tour& tour : tours) {
            tour.nodes.reserve(size);
            std::vector<unsigned char> visited(size, 0);
            std::vector<std::size_t> available;
            std::vector<double> desirability;
            available.reserve(size);
            desirability.reserve(size);

            std::size_t current = startNode(rng);
            tour.nodes.push_back(current);
            visited[current] = 1;
            tour.cost = 0.0;

            while (tour.nodes.size() < size) {
                available.clear();
                for (std::size_t node : candidateLists[current]) {
                    if (!visited[node]) {
                        available.push_back(node);
                    }
                }
                if (available.empty()) {
                    for (std::size_t node = 0; node < size; ++node) {
                        if (!visited[node]) {
                            available.push_back(node);
                        }
                    }
                }

                desirability.resize(available.size());
                for (std::size_t i = 0; i < available.size(); ++i) {
                    const std::size_t next = available[i];
                    desirability[i] =
                        std::pow(pheromone[current * size + next],
                                 config_.alpha * control.exploitation) *
                        std::pow(heuristic[current * size + next],
                                 config_.beta * control.exploitation);
                }

                std::size_t selected = 0;
                const double exploitationProbability = std::clamp(
                    config_.exploitationProbability /
                        std::max(control.randomization, 1e-12),
                    0.0,
                    1.0);
                if (config_.variant == AntColonyVariant::AntColonySystem &&
                    uniform01(rng) < exploitationProbability) {
                    selected = static_cast<std::size_t>(std::distance(
                        desirability.begin(),
                        std::max_element(desirability.begin(), desirability.end())));
                } else {
                    std::discrete_distribution<std::size_t> choose(
                        desirability.begin(), desirability.end());
                    selected = choose(rng);
                }

                const std::size_t next = available[selected];
                tour.cost += graph.cost(current, next);
                if (config_.variant == AntColonyVariant::AntColonySystem) {
                    const std::size_t edge = current * size + next;
                    pheromone[edge] = (1.0 - config_.localEvaporation) * pheromone[edge] +
                                      config_.localEvaporation * config_.initialPheromone;
                    if (graph.symmetric()) {
                        pheromone[next * size + current] = pheromone[edge];
                    }
                }
                current = next;
                tour.nodes.push_back(current);
                visited[current] = 1;
            }
            tour.cost += graph.cost(tour.nodes.back(), tour.nodes.front());
            if (config_.variant == AntColonyVariant::AntColonySystem) {
                const std::size_t from = tour.nodes.back();
                const std::size_t to = tour.nodes.front();
                const std::size_t edge = from * size + to;
                pheromone[edge] = (1.0 - config_.localEvaporation) * pheromone[edge] +
                                  config_.localEvaporation * config_.initialPheromone;
                if (graph.symmetric()) {
                    pheromone[to * size + from] = pheromone[edge];
                }
            }
        }

        std::sort(tours.begin(), tours.end(), [](const Tour& a, const Tour& b) {
            return a.cost < b.cost;
        });
        const double previousBest = result.bestCost;
        if (tours.front().cost < result.bestCost) {
            result.bestCost = tours.front().cost;
            result.bestTour = tours.front().nodes;
            stagnant = 0;
        } else {
            ++stagnant;
        }
        if (std::isfinite(previousBest)) {
            previousImprovement = std::max(0.0, previousBest - result.bestCost) /
                                  std::max(1.0, std::abs(previousBest));
        }

        control = ga::metaheuristics::detail::controlSignal(
            config_.controller.get(),
            iteration,
            config_.iterations,
            routeDiversity(tours, size),
            previousImprovement,
            stagnant);
        const double evaporation = std::clamp(
            config_.evaporation * control.evaporation, 1e-6, 1.0 - 1e-6);
        if (config_.variant != AntColonyVariant::AntColonySystem) {
            for (std::size_t from = 0; from < size; ++from) {
                for (std::size_t to = 0; to < size; ++to) {
                    if (from != to) {
                        pheromone[from * size + to] *= 1.0 - evaporation;
                    }
                }
            }
        }

        Tour bestSoFar{result.bestTour, result.bestCost};
        switch (config_.variant) {
        case AntColonyVariant::AntSystem:
            for (const Tour& tour : tours) {
                depositTour(pheromone,
                            size,
                            graph.symmetric(),
                            tour,
                            config_.depositScale / tour.cost);
            }
            break;
        case AntColonyVariant::ElitistAntSystem:
            for (const Tour& tour : tours) {
                depositTour(pheromone,
                            size,
                            graph.symmetric(),
                            tour,
                            config_.depositScale / tour.cost);
            }
            depositTour(pheromone,
                        size,
                        graph.symmetric(),
                        bestSoFar,
                        config_.elitistWeight * config_.depositScale / bestSoFar.cost);
            break;
        case AntColonyVariant::RankBasedAntSystem: {
            const std::size_t ranks = std::min(config_.rankCount, tours.size());
            for (std::size_t rank = 0; rank + 1 < ranks; ++rank) {
                const double weight = static_cast<double>(ranks - rank - 1);
                depositTour(pheromone,
                            size,
                            graph.symmetric(),
                            tours[rank],
                            weight * config_.depositScale / tours[rank].cost);
            }
            depositTour(pheromone,
                        size,
                        graph.symmetric(),
                        bestSoFar,
                        static_cast<double>(ranks) * config_.depositScale /
                            bestSoFar.cost);
            break;
        }
        case AntColonyVariant::AntColonySystem:
            for (std::size_t i = 0; i < size; ++i) {
                const std::size_t from = bestSoFar.nodes[i];
                const std::size_t to = bestSoFar.nodes[(i + 1) % size];
                const double updated =
                    (1.0 - evaporation) * pheromone[from * size + to] +
                    evaporation * config_.depositScale / bestSoFar.cost;
                pheromone[from * size + to] = updated;
                if (graph.symmetric()) {
                    pheromone[to * size + from] = updated;
                }
            }
            break;
        case AntColonyVariant::MaxMinAntSystem: {
            depositTour(pheromone,
                        size,
                        graph.symmetric(),
                        iteration % 5 == 0 ? bestSoFar : tours.front(),
                        config_.depositScale /
                            (iteration % 5 == 0 ? bestSoFar.cost : tours.front().cost));
            const double maximum = config_.maxPheromone > 0.0
                                       ? config_.maxPheromone
                                       : config_.depositScale /
                                             (evaporation * result.bestCost);
            const double minimum = config_.minPheromone > 0.0
                                       ? config_.minPheromone
                                       : maximum / (2.0 * static_cast<double>(size));
            for (std::size_t from = 0; from < size; ++from) {
                for (std::size_t to = 0; to < size; ++to) {
                    if (from != to) {
                        double& value = pheromone[from * size + to];
                        value = std::clamp(value, minimum, maximum);
                    }
                }
            }
            break;
        }
        }

        result.bestCostHistory.push_back(result.bestCost);
        result.evaluations += config_.ants;
        result.iterations = iteration + 1;
    }
    return result;
}

} // namespace aco
} // namespace ga
