#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "ga/config.hpp"
#include "ga/shake/shake_types.hpp"

namespace ga {
namespace shake {

inline void uniformShake(std::vector<double>& sol, const ga::Bounds& b, double intensity, std::mt19937& rng) {
    const double span = b.upper - b.lower;
    std::uniform_real_distribution<double> d(-intensity * span, intensity * span);
    for (double& val : sol) {
        val = std::clamp(val + d(rng), b.lower, b.upper);
    }
}

inline void gaussianShake(std::vector<double>& sol, const ga::Bounds& b, double intensity, std::mt19937& rng) {
    const double span = b.upper - b.lower;
    std::normal_distribution<double> d(0.0, intensity * span);
    for (double& val : sol) {
        val = std::clamp(val + d(rng), b.lower, b.upper);
    }
}

inline void cauchyShake(std::vector<double>& sol, const ga::Bounds& b, double intensity, std::mt19937& rng) {
    const double span = b.upper - b.lower;
    std::cauchy_distribution<double> d(0.0, intensity * span);
    for (double& val : sol) {
        val = std::clamp(val + d(rng), b.lower, b.upper);
    }
}

inline void levyFlightShake(std::vector<double>& sol, const ga::Bounds& b, double intensity, double beta, std::mt19937& rng) {
    beta = std::clamp(beta, 1.0, 2.0);
    const double span = b.upper - b.lower;
    const double pi = 3.14159265358979323846;
    const double num = std::tgamma(1.0 + beta) * std::sin(pi * beta * 0.5);
    const double den = std::tgamma((1.0 + beta) * 0.5) * beta * std::pow(2.0, (beta - 1.0) * 0.5);
    const double sigma_u = std::pow(num / den, 1.0 / beta);

    std::normal_distribution<double> dist_u(0.0, sigma_u);
    std::normal_distribution<double> dist_v(0.0, 1.0);

    for (double& val : sol) {
        const double u = dist_u(rng);
        const double v = dist_v(rng);
        const double step = u / std::pow(std::abs(v) + 1e-12, 1.0 / beta);
        val = std::clamp(val + intensity * step * span, b.lower, b.upper);
    }
}

inline void oppositionShake(std::vector<double>& sol, const ga::Bounds& b) {
    for (double& val : sol) {
        val = std::clamp(b.lower + b.upper - val, b.lower, b.upper);
    }
}

inline void partialDimensionShake(std::vector<double>& sol, const ga::Bounds& b, double intensity, std::size_t dims, std::mt19937& rng) {
    if (sol.empty()) return;
    const double span = b.upper - b.lower;
    std::vector<std::size_t> indices(sol.size());
    for (std::size_t i = 0; i < sol.size(); ++i) indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), rng);
    const std::size_t count = std::min(dims, sol.size());
    for (std::size_t k = 0; k < count; ++k) {
        const std::size_t i = indices[k];
        std::cauchy_distribution<double> d(0.0, intensity * span);
        sol[i] = std::clamp(sol[i] + d(rng), b.lower, b.upper);
    }
}

inline void applyShake(std::vector<double>& sol, const ga::Bounds& b, const ShakeConfig& cfg, std::mt19937& rng) {
    switch (cfg.type) {
        case ShakeType::Uniform: uniformShake(sol, b, cfg.intensity, rng); break;
        case ShakeType::Gaussian: gaussianShake(sol, b, cfg.intensity, rng); break;
        case ShakeType::Cauchy: cauchyShake(sol, b, cfg.intensity, rng); break;
        case ShakeType::LevyFlight: levyFlightShake(sol, b, cfg.intensity, cfg.levyBeta, rng); break;
        case ShakeType::Opposition: oppositionShake(sol, b); break;
        case ShakeType::PartialDimension: partialDimensionShake(sol, b, cfg.intensity, cfg.partialDims, rng); break;
    }
}

} // namespace shake
} // namespace ga
