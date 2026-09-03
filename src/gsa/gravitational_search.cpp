#include "ga/gsa/gravitational_search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace ga {
namespace gsa {

GravitationalSearchOptimizer::GravitationalSearchOptimizer(GsaConfig config)
    : config_(std::move(config)) {
    ga::metaheuristics::detail::validateSearchConfig(config_.search);
    if (!std::isfinite(config_.gravitationalConstant) ||
        config_.gravitationalConstant <= 0.0 || !std::isfinite(config_.decay) ||
        config_.decay < 0.0 || !std::isfinite(config_.epsilon) ||
        config_.epsilon <= 0.0 || config_.finalEliteFraction <= 0.0 ||
        config_.finalEliteFraction > 1.0) {
        throw std::invalid_argument("invalid GSA configuration");
    }
}

ga::core::OptimizationResult GravitationalSearchOptimizer::optimize(
    const ga::Fitness& fitness,
    const ga::metaheuristics::SeedPopulation& seeds) {
    using namespace ga::metaheuristics;
    const auto& search = config_.search;
    std::mt19937 rng = detail::makeRng(search.seed);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    SeedPopulation positions = detail::makePopulation(search, seeds, rng);
    SeedPopulation velocities(
        search.populationSize, std::vector<double>(search.dimension, 0.0));
    SeedPopulation accelerations = velocities;
    std::vector<double> values = detail::evaluateBatch(positions, fitness, search.threads);
    std::vector<double> masses(search.populationSize, 0.0);
    std::vector<std::size_t> ranked(search.populationSize);
    std::iota(ranked.begin(), ranked.end(), 0);

    ga::core::OptimizationResult result;
    result.bestFitness = -std::numeric_limits<double>::infinity();
    std::size_t evaluations = search.populationSize;
    std::size_t stagnant = 0;
    double lastImprovement = 1.0;
    detail::updateResult(result, positions, values, evaluations);

    for (std::size_t iteration = 0; iteration < search.iterations; ++iteration) {
        const double previousBest = result.bestFitness;
        const auto minmax = std::minmax_element(values.begin(), values.end());
        const double worst = *minmax.first;
        const double best = *minmax.second;
        const double spread = best - worst;

        if (spread <= config_.epsilon) {
            std::fill(masses.begin(), masses.end(),
                      1.0 / static_cast<double>(masses.size()));
        } else {
            double massSum = 0.0;
            for (std::size_t i = 0; i < values.size(); ++i) {
                masses[i] = (values[i] - worst) / spread + config_.epsilon;
                massSum += masses[i];
            }
            for (double& mass : masses) {
                mass /= massSum;
            }
        }

        std::sort(ranked.begin(), ranked.end(), [&](std::size_t a, std::size_t b) {
            return values[a] > values[b];
        });
        const double progress = static_cast<double>(iteration) /
                                static_cast<double>(search.iterations);
        const double eliteFraction = 1.0 -
            progress * (1.0 - config_.finalEliteFraction);
        const std::size_t eliteCount = std::max<std::size_t>(
            1,
            static_cast<std::size_t>(
                std::ceil(eliteFraction * static_cast<double>(search.populationSize))));

        const ControlSignal signal = detail::controlSignal(
            config_.controller.get(),
            iteration,
            search.iterations,
            detail::normalizedDiversity(positions, search.bounds),
            lastImprovement,
            stagnant);
        const double gravity = config_.gravitationalConstant * signal.exploration *
                               std::exp(-config_.decay * progress);

        for (auto& acceleration : accelerations) {
            std::fill(acceleration.begin(), acceleration.end(), 0.0);
        }
        for (std::size_t i = 0; i < search.populationSize; ++i) {
            for (std::size_t rank = 0; rank < eliteCount; ++rank) {
                const std::size_t j = ranked[rank];
                if (i == j) {
                    continue;
                }
                double squaredDistance = 0.0;
                for (std::size_t d = 0; d < search.dimension; ++d) {
                    const double delta = positions[j][d] - positions[i][d];
                    squaredDistance += delta * delta;
                }
                const double inverseDistance =
                    1.0 / (std::sqrt(squaredDistance) + config_.epsilon);
                for (std::size_t d = 0; d < search.dimension; ++d) {
                    accelerations[i][d] += uniform01(rng) * gravity * masses[j] *
                                            (positions[j][d] - positions[i][d]) *
                                            inverseDistance;
                }
            }
        }

        for (std::size_t i = 0; i < search.populationSize; ++i) {
            for (std::size_t d = 0; d < search.dimension; ++d) {
                velocities[i][d] = uniform01(rng) * velocities[i][d] +
                                    signal.exploitation * accelerations[i][d];
                positions[i][d] = std::clamp(
                    positions[i][d] + velocities[i][d],
                    search.bounds.lower,
                    search.bounds.upper);
            }
        }

        values = detail::evaluateBatch(positions, fitness, search.threads);
        evaluations += search.populationSize;
        detail::updateResult(result, positions, values, evaluations);
        result.generations = iteration + 1;
        lastImprovement = detail::relativeImprovement(result.bestFitness, previousBest);
        stagnant = result.bestFitness > previousBest ? 0 : stagnant + 1;
    }
    return result;
}

} // namespace gsa
} // namespace ga
