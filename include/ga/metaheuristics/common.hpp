#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <future>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ga/config.hpp"
#include "ga/core/result.hpp"

namespace ga {
namespace metaheuristics {

using SeedPopulation = std::vector<std::vector<double>>;

struct SearchConfig {
    std::size_t populationSize = 50;
    std::size_t iterations = 100;
    std::size_t dimension = 10;
    ga::Bounds bounds{-5.12, 5.12};
    unsigned seed = 0;
    std::size_t threads = 1;
};

struct ProgressState {
    std::size_t iteration = 0;
    std::size_t maxIterations = 1;
    double normalizedDiversity = 0.0;
    double relativeImprovement = 0.0;
    double stagnation = 0.0;
};

struct ControlSignal {
    double exploration = 1.0;
    double exploitation = 1.0;
    double evaporation = 1.0;
    double randomization = 1.0;
};

class IAdaptiveController {
public:
    virtual ~IAdaptiveController() = default;
    virtual ControlSignal update(const ProgressState& state) const = 0;
};

class IContinuousOptimizer {
public:
    virtual ~IContinuousOptimizer() = default;
    virtual std::string name() const = 0;
    virtual ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const SeedPopulation& seeds = {}) = 0;
};

namespace detail {

inline void validateSearchConfig(const SearchConfig& cfg) {
    if (cfg.populationSize == 0) {
        throw std::invalid_argument("populationSize must be greater than zero");
    }
    if (cfg.iterations == 0) {
        throw std::invalid_argument("iterations must be greater than zero");
    }
    if (cfg.dimension == 0) {
        throw std::invalid_argument("dimension must be greater than zero");
    }
    if (!std::isfinite(cfg.bounds.lower) || !std::isfinite(cfg.bounds.upper) ||
        cfg.bounds.lower >= cfg.bounds.upper ||
        !std::isfinite(cfg.bounds.upper - cfg.bounds.lower)) {
        throw std::invalid_argument("bounds must be finite and lower < upper");
    }
    if (cfg.threads == 0) {
        throw std::invalid_argument("threads must be greater than zero");
    }
}

inline std::mt19937 makeRng(unsigned seed) {
    return seed == 0 ? std::mt19937{std::random_device{}()} : std::mt19937{seed};
}

inline void clampToBounds(std::vector<double>& value, const ga::Bounds& bounds) {
    for (double& x : value) {
        x = std::clamp(x, bounds.lower, bounds.upper);
    }
}

inline SeedPopulation makePopulation(const SearchConfig& cfg,
                                     const SeedPopulation& seeds,
                                     std::mt19937& rng) {
    validateSearchConfig(cfg);
    SeedPopulation population;
    population.reserve(cfg.populationSize);

    for (const auto& seed : seeds) {
        if (population.size() == cfg.populationSize) {
            break;
        }
        if (seed.size() != cfg.dimension) {
            throw std::invalid_argument("seed solution dimension does not match SearchConfig");
        }
        if (!std::all_of(seed.begin(), seed.end(), [](double value) {
                return std::isfinite(value);
            })) {
            throw std::invalid_argument("seed solutions must contain only finite values");
        }
        population.push_back(seed);
        clampToBounds(population.back(), cfg.bounds);
    }

    std::uniform_real_distribution<double> uniform(cfg.bounds.lower, cfg.bounds.upper);
    while (population.size() < cfg.populationSize) {
        std::vector<double> solution(cfg.dimension);
        for (double& x : solution) {
            x = uniform(rng);
        }
        population.push_back(std::move(solution));
    }
    return population;
}

inline std::vector<double> evaluateBatch(const SeedPopulation& population,
                                         const ga::Fitness& fitness,
                                         std::size_t threads) {
    if (!fitness) {
        throw std::invalid_argument("fitness callback is empty");
    }
    std::vector<double> values(population.size());
    if (population.empty()) {
        return values;
    }

    const std::size_t workerCount = std::min<std::size_t>(
        std::max<std::size_t>(1, threads), population.size());
    if (workerCount == 1) {
        for (std::size_t i = 0; i < population.size(); ++i) {
            values[i] = fitness(population[i]);
        }
    } else {
        const std::size_t block = (population.size() + workerCount - 1) / workerCount;
        std::vector<std::future<void>> tasks;
        tasks.reserve(workerCount);
        for (std::size_t worker = 0; worker < workerCount; ++worker) {
            const std::size_t begin = worker * block;
            const std::size_t end = std::min(population.size(), begin + block);
            if (begin >= end) {
                break;
            }
            tasks.emplace_back(std::async(std::launch::async, [&, begin, end] {
                for (std::size_t i = begin; i < end; ++i) {
                    values[i] = fitness(population[i]);
                }
            }));
        }
        for (auto& task : tasks) {
            task.get();
        }
    }
    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::domain_error("fitness callback returned a non-finite value");
        }
    }
    return values;
}

inline std::size_t bestIndex(const std::vector<double>& fitness) {
    if (fitness.empty()) {
        throw std::invalid_argument("cannot select from an empty fitness vector");
    }
    return static_cast<std::size_t>(
        std::distance(fitness.begin(), std::max_element(fitness.begin(), fitness.end())));
}

inline double mean(const std::vector<double>& values) {
    return values.empty()
               ? 0.0
               : std::accumulate(values.begin(), values.end(), 0.0) /
                     static_cast<double>(values.size());
}

inline double normalizedDiversity(const SeedPopulation& population,
                                  const ga::Bounds& bounds) {
    if (population.size() < 2 || population.front().empty()) {
        return 0.0;
    }
    const std::size_t dimension = population.front().size();
    std::vector<double> centroid(dimension, 0.0);
    for (const auto& solution : population) {
        for (std::size_t d = 0; d < dimension; ++d) {
            centroid[d] += solution[d];
        }
    }
    for (double& value : centroid) {
        value /= static_cast<double>(population.size());
    }

    double squaredDistance = 0.0;
    for (const auto& solution : population) {
        for (std::size_t d = 0; d < dimension; ++d) {
            const double delta = solution[d] - centroid[d];
            squaredDistance += delta * delta;
        }
    }
    const double rms = std::sqrt(
        squaredDistance / static_cast<double>(population.size() * dimension));
    return std::clamp(rms / (bounds.upper - bounds.lower), 0.0, 1.0);
}

inline double relativeImprovement(double current, double previous) {
    if (!std::isfinite(previous)) {
        return 1.0;
    }
    return std::max(0.0, current - previous) /
           std::max(1.0, std::abs(previous));
}

inline ControlSignal controlSignal(const IAdaptiveController* controller,
                                   std::size_t iteration,
                                   std::size_t maxIterations,
                                   double diversity,
                                   double improvement,
                                   std::size_t stagnantIterations) {
    if (controller == nullptr) {
        return {};
    }
    ProgressState state;
    state.iteration = iteration;
    state.maxIterations = maxIterations;
    state.normalizedDiversity = std::clamp(diversity, 0.0, 1.0);
    state.relativeImprovement = std::clamp(improvement, 0.0, 1.0);
    state.stagnation = maxIterations == 0
                           ? 0.0
                           : std::clamp(static_cast<double>(stagnantIterations) /
                                            static_cast<double>(maxIterations),
                                        0.0,
                                        1.0);
    ControlSignal signal = controller->update(state);
    auto sanitize = [](double value) {
        return std::isfinite(value) && value > 0.0
                   ? std::clamp(value, 0.1, 10.0)
                   : 1.0;
    };
    signal.exploration = sanitize(signal.exploration);
    signal.exploitation = sanitize(signal.exploitation);
    signal.evaporation = sanitize(signal.evaporation);
    signal.randomization = sanitize(signal.randomization);
    return signal;
}

inline void updateResult(ga::core::OptimizationResult& result,
                         const SeedPopulation& population,
                         const std::vector<double>& fitness,
                         std::size_t evaluations) {
    const std::size_t index = bestIndex(fitness);
    if (result.bestSolution.empty() || fitness[index] > result.bestFitness) {
        result.bestSolution = population[index];
        result.bestFitness = fitness[index];
    }
    result.bestHistory.push_back(result.bestFitness);
    result.avgHistory.push_back(mean(fitness));
    result.evaluations = evaluations;
}

} // namespace detail
} // namespace metaheuristics
} // namespace ga
