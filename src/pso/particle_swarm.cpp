#include "ga/pso/particle_swarm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace ga {
namespace pso {
namespace {

void validate(const PsoConfig& config) {
    ga::metaheuristics::detail::validateSearchConfig(config.search);
    if (!std::isfinite(config.inertia) || !std::isfinite(config.cognitive) ||
        !std::isfinite(config.social) || !std::isfinite(config.constriction) ||
        !std::isfinite(config.velocityClamp) ||
        !std::isfinite(config.binaryVelocityClamp) ||
        !std::isfinite(config.quantumBeta) ||
        config.inertia < 0.0 || config.cognitive < 0.0 || config.social < 0.0 ||
        config.constriction <= 0.0 || config.velocityClamp <= 0.0 ||
        config.binaryVelocityClamp <= 0.0 ||
        config.quantumBeta <= 0.0) {
        throw std::invalid_argument("PSO coefficients must be non-negative and scales positive");
    }
    if (config.neighborhoodRadius == 0) {
        throw std::invalid_argument("neighborhoodRadius must be greater than zero");
    }
}

const char* variantName(PsoVariant variant) {
    switch (variant) {
    case PsoVariant::GlobalBest: return "PSO-global-best";
    case PsoVariant::LocalBest: return "PSO-local-best";
    case PsoVariant::Constriction: return "PSO-constriction";
    case PsoVariant::BareBones: return "PSO-bare-bones";
    case PsoVariant::FullyInformed: return "PSO-fully-informed";
    case PsoVariant::QuantumBehaved: return "PSO-quantum-behaved";
    case PsoVariant::Binary: return "PSO-binary";
    }
    return "PSO";
}

std::size_t localBestIndex(std::size_t particle,
                           std::size_t radius,
                           const std::vector<double>& personalFitness) {
    const std::size_t count = personalFitness.size();
    std::size_t best = particle;
    for (std::size_t offset = 1; offset <= std::min(radius, count - 1); ++offset) {
        const std::size_t left = (particle + count - offset) % count;
        const std::size_t right = (particle + offset) % count;
        if (personalFitness[left] > personalFitness[best]) {
            best = left;
        }
        if (personalFitness[right] > personalFitness[best]) {
            best = right;
        }
    }
    return best;
}

} // namespace

ParticleSwarmOptimizer::ParticleSwarmOptimizer(PsoConfig config)
    : config_(std::move(config)) {
    validate(config_);
}

std::string ParticleSwarmOptimizer::name() const {
    return variantName(config_.variant);
}

ga::core::OptimizationResult ParticleSwarmOptimizer::optimize(
    const ga::Fitness& fitness,
    const ga::metaheuristics::SeedPopulation& seeds) {
    using namespace ga::metaheuristics;
    const auto& search = config_.search;
    std::mt19937 rng = detail::makeRng(search.seed);
    SeedPopulation positions = detail::makePopulation(search, seeds, rng);
    const std::size_t particles = positions.size();
    const std::size_t dimension = search.dimension;
    const double range = search.bounds.upper - search.bounds.lower;
    const double maxVelocity = config_.variant == PsoVariant::Binary
                                   ? config_.binaryVelocityClamp
                                   : config_.velocityClamp * range;

    if (config_.variant == PsoVariant::Binary) {
        std::bernoulli_distribution bit(0.5);
        const std::size_t seeded = std::min(seeds.size(), particles);
        for (std::size_t i = 0; i < particles; ++i) {
            for (double& value : positions[i]) {
                value = i < seeded ? (value >= 0.5 ? 1.0 : 0.0)
                                   : (bit(rng) ? 1.0 : 0.0);
            }
        }
    }

    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    std::uniform_real_distribution<double> initialVelocity(-maxVelocity, maxVelocity);
    SeedPopulation velocities(particles, std::vector<double>(dimension, 0.0));
    for (auto& velocity : velocities) {
        for (double& value : velocity) {
            value = initialVelocity(rng);
        }
    }

    std::vector<double> values = detail::evaluateBatch(positions, fitness, search.threads);
    SeedPopulation personalBest = positions;
    std::vector<double> personalFitness = values;
    std::size_t globalIndex = detail::bestIndex(personalFitness);
    std::vector<double> globalBest = personalBest[globalIndex];
    double globalFitness = personalFitness[globalIndex];

    ga::core::OptimizationResult result;
    result.bestFitness = -std::numeric_limits<double>::infinity();
    std::size_t evaluations = particles;
    std::size_t stagnant = 0;
    double lastImprovement = 1.0;
    detail::updateResult(result, positions, values, evaluations);

    for (std::size_t iteration = 0; iteration < search.iterations; ++iteration) {
        const double previousBest = globalFitness;
        const double diversity = detail::normalizedDiversity(positions, search.bounds);
        const ControlSignal signal = detail::controlSignal(
            config_.controller.get(),
            iteration,
            search.iterations,
            diversity,
            lastImprovement,
            stagnant);

        std::vector<double> meanPersonalBest(dimension, 0.0);
        if (config_.variant == PsoVariant::QuantumBehaved) {
            for (const auto& best : personalBest) {
                for (std::size_t d = 0; d < dimension; ++d) {
                    meanPersonalBest[d] += best[d];
                }
            }
            for (double& value : meanPersonalBest) {
                value /= static_cast<double>(particles);
            }
        }

        for (std::size_t i = 0; i < particles; ++i) {
            std::size_t guideIndex = globalIndex;
            if (config_.variant == PsoVariant::LocalBest) {
                guideIndex = localBestIndex(i, config_.neighborhoodRadius, personalFitness);
            }
            const auto& guide = personalBest[guideIndex];

            for (std::size_t d = 0; d < dimension; ++d) {
                const double r1 = uniform01(rng);
                const double r2 = uniform01(rng);
                double next = positions[i][d];

                switch (config_.variant) {
                case PsoVariant::BareBones: {
                    const double mean = 0.5 * (personalBest[i][d] + guide[d]);
                    const double sigma = std::max(
                        std::abs(personalBest[i][d] - guide[d]) * signal.exploration,
                        range * 1e-12);
                    std::normal_distribution<double> normal(mean, sigma);
                    next = normal(rng);
                    break;
                }
                case PsoVariant::FullyInformed: {
                    double informed = 0.0;
                    for (std::size_t j = 0; j < particles; ++j) {
                        informed += uniform01(rng) * (personalBest[j][d] - positions[i][d]);
                    }
                    informed /= static_cast<double>(particles);
                    velocities[i][d] = config_.inertia * signal.exploration * velocities[i][d] +
                                       (config_.cognitive + config_.social) *
                                           signal.exploitation * informed;
                    velocities[i][d] = std::clamp(velocities[i][d], -maxVelocity, maxVelocity);
                    next += velocities[i][d];
                    break;
                }
                case PsoVariant::QuantumBehaved: {
                    const double phi = uniform01(rng);
                    const double attractor = phi * personalBest[i][d] + (1.0 - phi) * guide[d];
                    const double u = std::max(uniform01(rng), std::numeric_limits<double>::min());
                    const double direction = uniform01(rng) < 0.5 ? -1.0 : 1.0;
                    next = attractor + direction * config_.quantumBeta * signal.exploration *
                                           std::abs(meanPersonalBest[d] - positions[i][d]) *
                                           std::log(1.0 / u);
                    break;
                }
                default: {
                    const double cognitive = config_.cognitive * signal.exploitation * r1 *
                                             (personalBest[i][d] - positions[i][d]);
                    const double social = config_.social * signal.exploitation * r2 *
                                          (guide[d] - positions[i][d]);
                    const double inertiaCoefficient =
                        config_.variant == PsoVariant::Constriction ? 1.0 : config_.inertia;
                    const double inertia = inertiaCoefficient * signal.exploration *
                                           velocities[i][d];
                    velocities[i][d] = inertia + cognitive + social;
                    if (config_.variant == PsoVariant::Constriction) {
                        velocities[i][d] *= config_.constriction;
                    }
                    velocities[i][d] = std::clamp(velocities[i][d], -maxVelocity, maxVelocity);
                    if (config_.variant == PsoVariant::Binary) {
                        const double probability = 1.0 / (1.0 + std::exp(-velocities[i][d]));
                        next = uniform01(rng) < probability ? 1.0 : 0.0;
                    } else {
                        next += velocities[i][d];
                    }
                    break;
                }
                }

                positions[i][d] = config_.variant == PsoVariant::Binary
                                      ? next
                                      : std::clamp(next, search.bounds.lower, search.bounds.upper);
            }
        }

        values = detail::evaluateBatch(positions, fitness, search.threads);
        evaluations += particles;
        for (std::size_t i = 0; i < particles; ++i) {
            if (values[i] > personalFitness[i]) {
                personalFitness[i] = values[i];
                personalBest[i] = positions[i];
            }
        }
        globalIndex = detail::bestIndex(personalFitness);
        globalBest = personalBest[globalIndex];
        globalFitness = personalFitness[globalIndex];
        lastImprovement = detail::relativeImprovement(globalFitness, previousBest);
        stagnant = globalFitness > previousBest ? 0 : stagnant + 1;

        detail::updateResult(result, positions, values, evaluations);
        result.generations = iteration + 1;
    }

    result.bestSolution = std::move(globalBest);
    result.bestFitness = globalFitness;
    return result;
}

} // namespace pso
} // namespace ga
