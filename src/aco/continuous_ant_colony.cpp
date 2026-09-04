#include "ga/aco/continuous_ant_colony.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace ga {
namespace aco {

ContinuousAntColonyOptimizer::ContinuousAntColonyOptimizer(AcorConfig config)
    : config_(std::move(config)) {
    ga::metaheuristics::detail::validateSearchConfig(config_.search);
    if (config_.archiveSize < 2 || config_.sampleCount == 0 ||
        !std::isfinite(config_.locality) || config_.locality <= 0.0 ||
        !std::isfinite(config_.convergenceSpeed) || config_.convergenceSpeed <= 0.0) {
        throw std::invalid_argument("invalid ACOR configuration");
    }
}

ga::core::OptimizationResult ContinuousAntColonyOptimizer::optimize(
    const ga::Fitness& fitness,
    const ga::metaheuristics::SeedPopulation& seeds) {
    using namespace ga::metaheuristics;
    SearchConfig archiveConfig = config_.search;
    archiveConfig.populationSize = config_.archiveSize;
    std::mt19937 rng = detail::makeRng(archiveConfig.seed);
    SeedPopulation archive = detail::makePopulation(archiveConfig, seeds, rng);
    std::vector<double> archiveFitness =
        detail::evaluateBatch(archive, fitness, archiveConfig.threads);

    auto sortArchive = [&] {
        std::vector<std::size_t> order(archive.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return archiveFitness[a] > archiveFitness[b];
        });
        SeedPopulation sortedArchive;
        std::vector<double> sortedFitness;
        sortedArchive.reserve(archive.size());
        sortedFitness.reserve(archive.size());
        for (std::size_t index : order) {
            sortedArchive.push_back(std::move(archive[index]));
            sortedFitness.push_back(archiveFitness[index]);
        }
        archive = std::move(sortedArchive);
        archiveFitness = std::move(sortedFitness);
    };
    sortArchive();

    const double pi = std::acos(-1.0);
    std::vector<double> weights(config_.archiveSize, 0.0);
    for (std::size_t rank = 0; rank < config_.archiveSize; ++rank) {
        const double scaled = static_cast<double>(rank) /
                              (config_.locality * static_cast<double>(config_.archiveSize));
        weights[rank] = std::exp(-0.5 * scaled * scaled) /
                        (config_.locality * static_cast<double>(config_.archiveSize) *
                         std::sqrt(2.0 * pi));
    }
    std::discrete_distribution<std::size_t> selectKernel(weights.begin(), weights.end());

    ga::core::OptimizationResult result;
    result.bestFitness = -std::numeric_limits<double>::infinity();
    std::size_t evaluations = config_.archiveSize;
    std::size_t stagnant = 0;
    double lastImprovement = 1.0;
    detail::updateResult(result, archive, archiveFitness, evaluations);

    for (std::size_t iteration = 0; iteration < config_.search.iterations; ++iteration) {
        const double previousBest = result.bestFitness;
        const ControlSignal signal = detail::controlSignal(
            config_.controller.get(),
            iteration,
            config_.search.iterations,
            detail::normalizedDiversity(archive, config_.search.bounds),
            lastImprovement,
            stagnant);

        SeedPopulation samples(config_.sampleCount,
                               std::vector<double>(config_.search.dimension));
        for (auto& sample : samples) {
            const std::size_t kernel = selectKernel(rng);
            for (std::size_t d = 0; d < config_.search.dimension; ++d) {
                double absoluteDeviation = 0.0;
                for (std::size_t j = 0; j < config_.archiveSize; ++j) {
                    if (j != kernel) {
                        absoluteDeviation += std::abs(archive[j][d] - archive[kernel][d]);
                    }
                }
                double sigma = config_.convergenceSpeed * signal.exploration *
                               absoluteDeviation /
                               static_cast<double>(config_.archiveSize - 1);
                sigma = std::max(sigma,
                                 (config_.search.bounds.upper - config_.search.bounds.lower) *
                                     1e-12);
                std::normal_distribution<double> normal(archive[kernel][d], sigma);
                sample[d] = std::clamp(normal(rng),
                                       config_.search.bounds.lower,
                                       config_.search.bounds.upper);
            }
        }

        std::vector<double> sampleFitness =
            detail::evaluateBatch(samples, fitness, config_.search.threads);
        evaluations += config_.sampleCount;
        archive.reserve(config_.archiveSize + config_.sampleCount);
        archiveFitness.reserve(config_.archiveSize + config_.sampleCount);
        for (std::size_t i = 0; i < samples.size(); ++i) {
            archive.push_back(std::move(samples[i]));
            archiveFitness.push_back(sampleFitness[i]);
        }
        sortArchive();
        archive.resize(config_.archiveSize);
        archiveFitness.resize(config_.archiveSize);

        detail::updateResult(result, archive, archiveFitness, evaluations);
        result.generations = iteration + 1;
        lastImprovement = detail::relativeImprovement(result.bestFitness, previousBest);
        stagnant = result.bestFitness > previousBest ? 0 : stagnant + 1;
    }
    return result;
}

} // namespace aco
} // namespace ga
