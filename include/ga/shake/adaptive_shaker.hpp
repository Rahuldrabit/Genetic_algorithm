#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "ga/config.hpp"
#include "ga/shake/shake_types.hpp"
#include "ga/shake/shake_operators.hpp"

namespace ga {
namespace shake {

struct AdaptiveShakerConfig {
    ShakeConfig shake;
    std::size_t stagnationThreshold = 10;
    double diversityThreshold = 0.05;
    double fractionToShake = 0.5;
};

class AdaptiveSwarmShaker {
public:
    explicit AdaptiveSwarmShaker(AdaptiveShakerConfig config = {})
        : config_(config) {}

    bool checkAndShake(std::vector<std::vector<double>>& pop,
                       std::vector<double>& fits,
                       std::size_t bestIdx,
                       const ga::Bounds& bounds,
                       const ga::Fitness& fitnessFunc,
                       std::mt19937& rng,
                       double forcedIntensity = -1.0) {
        if (pop.empty() || fits.empty() || bestIdx >= pop.size()) return false;
        const std::size_t n = pop.size();
        const std::size_t dim = pop.front().size();

        const double currentBestFit = fits[bestIdx];
        if (currentBestFit > lastBestFitness_ + 1e-8) {
            lastBestFitness_ = currentBestFit;
            stagnationCount_ = 0;
        } else {
            stagnationCount_++;
        }

        // Compute normalized diversity
        std::vector<double> mean(dim, 0.0);
        for (const auto& ind : pop) {
            for (std::size_t d = 0; d < dim; ++d) mean[d] += ind[d];
        }
        for (std::size_t d = 0; d < dim; ++d) mean[d] /= static_cast<double>(n);

        const double s = bounds.upper - bounds.lower;
        const double diag = std::max(s * std::sqrt(static_cast<double>(dim)), 1e-12);
        double totalDist = 0.0;

        for (const auto& ind : pop) {
            double dSq = 0.0;
            for (std::size_t d = 0; d < dim; ++d) {
                const double diff = ind[d] - mean[d];
                dSq += diff * diff;
            }
            totalDist += std::sqrt(dSq);
        }
        lastDiversity_ = (totalDist / static_cast<double>(n)) / diag;

        const bool trigger = (stagnationCount_ >= config_.stagnationThreshold) ||
                             (lastDiversity_ < config_.diversityThreshold) ||
                             (forcedIntensity > 0.0);

        if (!trigger) return false;

        ShakeConfig effShake = config_.shake;
        if (forcedIntensity > 0.0) effShake.intensity = forcedIntensity;

        const std::size_t numToShake = std::max<std::size_t>(1, static_cast<std::size_t>(n * config_.fractionToShake));
        for (std::size_t k = 0; k < numToShake; ++k) {
            std::uniform_int_distribution<std::size_t> distIdx(0, n - 1);
            std::size_t target = distIdx(rng);
            if (target == bestIdx) target = (target + 1) % n;

            std::vector<double> candidate = pop[target];
            applyShake(candidate, bounds, effShake, rng);
            if (fitnessFunc && effShake.retainBest) {
                const double newFit = fitnessFunc(candidate);
                if (newFit >= fits[target]) {
                    pop[target] = std::move(candidate);
                    fits[target] = newFit;
                }
            } else {
                pop[target] = std::move(candidate);
                if (fitnessFunc) fits[target] = fitnessFunc(pop[target]);
            }
        }
        stagnationCount_ = 0;
        return true;
    }

    void reset() noexcept {
        stagnationCount_ = 0;
        lastBestFitness_ = -1e300;
        lastDiversity_ = 1.0;
    }

    std::size_t stagnationCount() const noexcept { return stagnationCount_; }
    double lastDiversity() const noexcept { return lastDiversity_; }
    const AdaptiveShakerConfig& config() const noexcept { return config_; }

private:
    AdaptiveShakerConfig config_;
    std::size_t stagnationCount_ = 0;
    double lastBestFitness_ = -1e300;
    double lastDiversity_ = 1.0;
};

} // namespace shake
} // namespace ga
