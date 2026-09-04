#pragma once

#include <cstddef>
#include <vector>

namespace ga {
namespace fuzzy {

struct FuzzyCMeansConfig {
    std::size_t clusters = 3;
    std::size_t maxIterations = 300;
    double fuzziness = 2.0;
    double tolerance = 1e-6;
    unsigned seed = 0;
};

struct FuzzyCMeansResult {
    std::vector<std::vector<double>> centers;
    std::vector<std::vector<double>> membership;
    std::vector<double> objectiveHistory;
    std::size_t iterations = 0;
    bool converged = false;

    std::vector<std::size_t> labels() const;
};

class FuzzyCMeans {
public:
    explicit FuzzyCMeans(FuzzyCMeansConfig config = {});

    FuzzyCMeansResult fit(const std::vector<std::vector<double>>& data) const;
    const FuzzyCMeansConfig& config() const noexcept { return config_; }

private:
    FuzzyCMeansConfig config_;
};

} // namespace fuzzy
} // namespace ga
