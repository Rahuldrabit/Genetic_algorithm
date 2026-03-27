#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <utility>
#include <vector>

namespace ga {
namespace es {

// Diagonal-CMA-ES style optimizer (practical approximation; covariance kept diagonal).
struct CmaEsConfig {
    std::size_t populationSize = 32;
    std::size_t generations = 120;
    std::size_t dimension = 10;
    double lower = -5.0;
    double upper = 5.0;
    double sigma = 0.3;
    unsigned seed = 0;
};

struct CmaEsResult {
    std::vector<double> best;
    double bestFitness = -1e300;
    std::vector<double> history;
};

class DiagonalCmaEs {
public:
    using Fitness = std::function<double(const std::vector<double>&)>;

    explicit DiagonalCmaEs(CmaEsConfig cfg)
        : cfg_(std::move(cfg)) {}

    CmaEsResult run(const Fitness& fitness) const {
        std::mt19937 rng(cfg_.seed == 0 ? std::random_device{}() : cfg_.seed);
        std::uniform_real_distribution<double> init(cfg_.lower, cfg_.upper);
        std::normal_distribution<double> stdnorm(0.0, 1.0);

        std::vector<double> mean(cfg_.dimension, 0.0);
        std::vector<double> diagVar(cfg_.dimension, cfg_.sigma * cfg_.sigma);
        for (double& v : mean) {
            v = init(rng);
        }

        const std::size_t mu = std::max<std::size_t>(1, cfg_.populationSize / 2);
        const double w = 1.0 / static_cast<double>(mu);

        struct Candidate {
            std::vector<double> x;
            double f = -1e300;
        };

        CmaEsResult out;
        out.history.reserve(cfg_.generations + 1);

        for (std::size_t gen = 0; gen < cfg_.generations; ++gen) {
            std::vector<Candidate> pop;
            pop.reserve(cfg_.populationSize);

            for (std::size_t i = 0; i < cfg_.populationSize; ++i) {
                Candidate c;
                c.x.resize(cfg_.dimension);
                for (std::size_t d = 0; d < cfg_.dimension; ++d) {
                    const double s = std::sqrt(std::max(1e-12, diagVar[d]));
                    c.x[d] = std::clamp(mean[d] + s * stdnorm(rng), cfg_.lower, cfg_.upper);
                }
                c.f = fitness(c.x);
                pop.push_back(std::move(c));
            }

            std::sort(pop.begin(), pop.end(), [](const Candidate& a, const Candidate& b) {
                return a.f > b.f;
            });

            if (pop.front().f > out.bestFitness) {
                out.bestFitness = pop.front().f;
                out.best = pop.front().x;
            }
            out.history.push_back(out.bestFitness);

            std::vector<double> newMean(cfg_.dimension, 0.0);
            for (std::size_t i = 0; i < mu; ++i) {
                for (std::size_t d = 0; d < cfg_.dimension; ++d) {
                    newMean[d] += w * pop[i].x[d];
                }
            }

            // Exponential moving update of diagonal covariance.
            for (std::size_t d = 0; d < cfg_.dimension; ++d) {
                double var = 0.0;
                for (std::size_t i = 0; i < mu; ++i) {
                    const double diff = pop[i].x[d] - newMean[d];
                    var += w * diff * diff;
                }
                diagVar[d] = 0.8 * diagVar[d] + 0.2 * std::max(1e-12, var);
            }

            mean = std::move(newMean);
        }

        return out;
    }

private:
    CmaEsConfig cfg_;
};

} // namespace es
} // namespace ga
