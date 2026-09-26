#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "ga/fuzzy/fuzzy_c_means.hpp"

namespace ga {
namespace fuzzy {

struct FcmSwarmMetrics {
    double partitionCoefficient = 1.0;
    double partitionEntropy = 0.0;
    double compactness = 0.0;
    double dominantCrowding = 0.0;
    std::vector<std::vector<double>> clusterCenters;
};

class FcmSwarmAnalyzer {
public:
    explicit FcmSwarmAnalyzer(FuzzyCMeansConfig config = {})
        : config_(config) {}

    FcmSwarmMetrics analyze(const std::vector<std::vector<double>>& population) const {
        FcmSwarmMetrics metrics;
        if (population.empty() || population.front().empty()) return metrics;
        const std::size_t n = population.size();

        FuzzyCMeans fcm(config_);
        auto res = fcm.fit(population);
        metrics.clusterCenters = res.centers;

        const std::size_t c = res.centers.size();
        if (c == 0 || n == 0) return metrics;

        double vpc = 0.0, vpe = 0.0;
        std::vector<std::size_t> clusterCounts(c, 0);

        for (std::size_t i = 0; i < n; ++i) {
            std::size_t bestC = 0;
            double bestU = -1.0;
            for (std::size_t j = 0; j < c; ++j) {
                const double u = std::clamp(res.membership[i][j], 0.0, 1.0);
                vpc += u * u;
                if (u > 1e-12) vpe -= u * std::log(u);
                if (u > bestU) { bestU = u; bestC = j; }
            }
            clusterCounts[bestC]++;
        }
        metrics.partitionCoefficient = vpc / static_cast<double>(n);
        metrics.partitionEntropy = vpe / static_cast<double>(n);

        std::size_t maxCount = 0;
        for (std::size_t count : clusterCounts) {
            if (count > maxCount) maxCount = count;
        }
        metrics.dominantCrowding = static_cast<double>(maxCount) / static_cast<double>(n);

        double totalDist = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double minDist = 1e18;
            for (std::size_t j = 0; j < c; ++j) {
                double d = 0.0;
                for (std::size_t dIdx = 0; dIdx < population[i].size(); ++dIdx) {
                    const double diff = population[i][dIdx] - res.centers[j][dIdx];
                    d += diff * diff;
                }
                if (d < minDist) minDist = d;
            }
            totalDist += std::sqrt(minDist);
        }
        metrics.compactness = totalDist / static_cast<double>(n);
        return metrics;
    }

    const FuzzyCMeansConfig& config() const noexcept { return config_; }

private:
    FuzzyCMeansConfig config_;
};

} // namespace fuzzy
} // namespace ga
