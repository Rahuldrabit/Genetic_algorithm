#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>
#include "ga/fuzzy/fuzzy_types.hpp"

namespace ga {
namespace fuzzy {

using MembershipCurve = std::function<double(double)>;

struct DefuzzificationConfig {
    DefuzzMethod method = DefuzzMethod::Centroid;
    std::size_t resolution = 200;
    double fallback = 0.0;
};

inline double defuzzifyWeightedAverage(const std::vector<double>& weights,
                                       const std::vector<double>& values,
                                       double fallback = 0.0) {
    double num = 0.0, denom = 0.0;
    const std::size_t n = std::min(weights.size(), values.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (weights[i] > 0.0) {
            num += weights[i] * values[i];
            denom += weights[i];
        }
    }
    return (denom > 1e-12) ? (num / denom) : fallback;
}

inline double defuzzify(const MembershipCurve& curve,
                        const Interval& domain,
                        DefuzzMethod method,
                        std::size_t resolution = 200,
                        double fallback = 0.0) {
    if (!curve || resolution < 2 || domain.max <= domain.min) return fallback;
    const double step = (domain.max - domain.min) / static_cast<double>(resolution);

    if (method == DefuzzMethod::WeightedAverage) return fallback;

    double num = 0.0, denom = 0.0;
    double maxMu = 0.0;
    std::vector<double> maxPositions;

    for (std::size_t i = 0; i <= resolution; ++i) {
        const double x = domain.min + i * step;
        const double mu = std::clamp(curve(x), 0.0, 1.0);
        num += x * mu;
        denom += mu;
        if (mu > maxMu + 1e-6) {
            maxMu = mu;
            maxPositions.clear();
            maxPositions.push_back(x);
        } else if (std::abs(mu - maxMu) <= 1e-6 && maxMu > 1e-6) {
            maxPositions.push_back(x);
        }
    }

    if (denom < 1e-12) return fallback;

    switch (method) {
        case DefuzzMethod::Centroid:
            return num / denom;
        case DefuzzMethod::Bisector: {
            const double halfArea = denom * 0.5;
            double cumArea = 0.0;
            for (std::size_t i = 0; i <= resolution; ++i) {
                const double x = domain.min + i * step;
                cumArea += std::clamp(curve(x), 0.0, 1.0);
                if (cumArea >= halfArea) return x;
            }
            return domain.center();
        }
        case DefuzzMethod::MOM: {
            if (maxPositions.empty()) return fallback;
            double sum = 0.0;
            for (double p : maxPositions) sum += p;
            return sum / static_cast<double>(maxPositions.size());
        }
        case DefuzzMethod::SOM:
            return maxPositions.empty() ? fallback : maxPositions.front();
        case DefuzzMethod::LOM:
            return maxPositions.empty() ? fallback : maxPositions.back();
        default:
            return num / denom;
    }
}

} // namespace fuzzy
} // namespace ga
