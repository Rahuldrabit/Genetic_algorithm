#include "ga/fuzzy/fuzzy_c_means.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace ga {
namespace fuzzy {
namespace {

double squaredDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double delta = a[i] - b[i];
        sum += delta * delta;
    }
    return sum;
}

void validateData(const std::vector<std::vector<double>>& data,
                  const FuzzyCMeansConfig& config) {
    if (data.empty() || data.front().empty()) {
        throw std::invalid_argument("fuzzy C-means data must be non-empty");
    }
    const std::size_t dimension = data.front().size();
    for (const auto& row : data) {
        if (row.size() != dimension) {
            throw std::invalid_argument("fuzzy C-means data must be rectangular");
        }
        for (double value : row) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("fuzzy C-means data must be finite");
            }
        }
    }
    if (config.clusters == 0 || config.clusters > data.size()) {
        throw std::invalid_argument("clusters must be in [1, sample count]");
    }
    if (config.maxIterations == 0) {
        throw std::invalid_argument("maxIterations must be greater than zero");
    }
    if (!std::isfinite(config.fuzziness) || config.fuzziness <= 1.0) {
        throw std::invalid_argument("fuzziness must be finite and greater than one");
    }
    if (!std::isfinite(config.tolerance) || config.tolerance <= 0.0) {
        throw std::invalid_argument("tolerance must be finite and positive");
    }
}

} // namespace

FuzzyCMeans::FuzzyCMeans(FuzzyCMeansConfig config) : config_(std::move(config)) {}

std::vector<std::size_t> FuzzyCMeansResult::labels() const {
    std::vector<std::size_t> out;
    out.reserve(membership.size());
    for (const auto& row : membership) {
        if (row.empty()) {
            out.push_back(0);
        } else {
            out.push_back(static_cast<std::size_t>(
                std::distance(row.begin(), std::max_element(row.begin(), row.end()))));
        }
    }
    return out;
}

FuzzyCMeansResult FuzzyCMeans::fit(
    const std::vector<std::vector<double>>& data) const {
    validateData(data, config_);
    const std::size_t samples = data.size();
    const std::size_t dimension = data.front().size();
    const std::size_t clusters = config_.clusters;

    std::mt19937 rng(config_.seed == 0 ? std::random_device{}() : config_.seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    FuzzyCMeansResult result;
    result.membership.assign(samples, std::vector<double>(clusters));
    result.centers.assign(clusters, std::vector<double>(dimension, 0.0));
    result.objectiveHistory.reserve(config_.maxIterations);

    for (auto& row : result.membership) {
        double total = 0.0;
        for (double& membership : row) {
            membership = std::max(uniform(rng), std::numeric_limits<double>::epsilon());
            total += membership;
        }
        for (double& membership : row) {
            membership /= total;
        }
    }

    std::vector<std::vector<double>> nextMembership(
        samples, std::vector<double>(clusters, 0.0));
    std::vector<double> distances(clusters, 0.0);
    const double exponent = 1.0 / (config_.fuzziness - 1.0);

    auto updateCenters = [&] {
        for (std::size_t c = 0; c < clusters; ++c) {
            std::fill(result.centers[c].begin(), result.centers[c].end(), 0.0);
            double denominator = 0.0;
            for (std::size_t i = 0; i < samples; ++i) {
                const double weight = std::pow(result.membership[i][c], config_.fuzziness);
                denominator += weight;
                for (std::size_t d = 0; d < dimension; ++d) {
                    result.centers[c][d] += weight * data[i][d];
                }
            }
            const double safeDenominator = std::max(
                denominator, std::numeric_limits<double>::min());
            for (double& value : result.centers[c]) {
                value /= safeDenominator;
            }
        }
    };

    for (std::size_t iteration = 0; iteration < config_.maxIterations; ++iteration) {
        updateCenters();

        double objective = 0.0;
        double maxChange = 0.0;
        for (std::size_t i = 0; i < samples; ++i) {
            std::size_t coincident = clusters;
            for (std::size_t c = 0; c < clusters; ++c) {
                distances[c] = squaredDistance(data[i], result.centers[c]);
                objective += std::pow(result.membership[i][c], config_.fuzziness) *
                             distances[c];
                if (distances[c] <= std::numeric_limits<double>::epsilon()) {
                    coincident = c;
                }
            }

            if (coincident < clusters) {
                std::fill(nextMembership[i].begin(), nextMembership[i].end(), 0.0);
                nextMembership[i][coincident] = 1.0;
            } else {
                for (std::size_t c = 0; c < clusters; ++c) {
                    double denominator = 0.0;
                    for (std::size_t k = 0; k < clusters; ++k) {
                        denominator += std::pow(distances[c] / distances[k], exponent);
                    }
                    nextMembership[i][c] = 1.0 / denominator;
                }
            }

            for (std::size_t c = 0; c < clusters; ++c) {
                maxChange = std::max(
                    maxChange,
                    std::abs(nextMembership[i][c] - result.membership[i][c]));
            }
        }

        result.membership.swap(nextMembership);
        result.objectiveHistory.push_back(objective);
        result.iterations = iteration + 1;
        if (maxChange <= config_.tolerance) {
            result.converged = true;
            break;
        }
    }

    // Membership is updated after centers inside the loop. Recompute once so
    // the returned centers correspond exactly to the returned memberships.
    updateCenters();

    return result;
}

} // namespace fuzzy
} // namespace ga
