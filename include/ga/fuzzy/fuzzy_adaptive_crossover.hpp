#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "crossover/base_crossover.h"
#include "ga/fuzzy/mamdani_system.hpp"

namespace ga {
namespace fuzzy {

enum class AdaptiveCrossoverMode { AutoFuzzy, BlendBLX, SimulatedBinary, Arithmetic };

struct FuzzyAdaptiveCrossoverConfig {
    AdaptiveCrossoverMode mode = AdaptiveCrossoverMode::AutoFuzzy;
    double searchDiameter = 10.0;
    double defaultAlpha = 0.5;
    double defaultEtaC = 2.0;
    unsigned seed = 0;
};

class FuzzyAdaptiveCrossover : public CrossoverOperator {
public:
    explicit FuzzyAdaptiveCrossover(FuzzyAdaptiveCrossoverConfig config = {})
        : config_(config), rng_(config.seed == 0 ? std::random_device{}() : config.seed) {
        initFuzzyEngine();
    }

    std::pair<RealVector, RealVector> crossover(
        const RealVector& parent1, const RealVector& parent2) override {
        operation_count++;
        const std::size_t n = std::min(parent1.size(), parent2.size());
        if (n == 0) return {parent1, parent2};

        std::map<std::string, double> in = {{"diversity", populationDiversity_}, {"generation", relativeGeneration_}};
        const double alpha = fis_.evaluateSingle("alpha", in, config_.defaultAlpha);
        const double eta = fis_.evaluateSingle("eta", in, config_.defaultEtaC);

        RealVector child1(n), child2(n);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        AdaptiveCrossoverMode effMode = config_.mode;
        if (effMode == AdaptiveCrossoverMode::AutoFuzzy) {
            effMode = (populationDiversity_ < 0.3) ? AdaptiveCrossoverMode::BlendBLX : AdaptiveCrossoverMode::SimulatedBinary;
        }

        for (std::size_t i = 0; i < n; ++i) {
            const double y1 = std::min(parent1[i], parent2[i]);
            const double y2 = std::max(parent1[i], parent2[i]);
            const double d = y2 - y1;

            if (effMode == AdaptiveCrossoverMode::BlendBLX) {
                const double lower = y1 - alpha * d;
                const double upper = y2 + alpha * d;
                std::uniform_real_distribution<double> dist(lower, upper);
                child1[i] = dist(rng_);
                child2[i] = dist(rng_);
            } else if (effMode == AdaptiveCrossoverMode::SimulatedBinary) {
                const double u = u01(rng_);
                double beta = (u <= 0.5) ? std::pow(2.0 * u, 1.0 / (eta + 1.0))
                                         : std::pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (eta + 1.0));
                child1[i] = 0.5 * ((1.0 + beta) * parent1[i] + (1.0 - beta) * parent2[i]);
                child2[i] = 0.5 * ((1.0 - beta) * parent1[i] + (1.0 + beta) * parent2[i]);
            } else {
                const double w = u01(rng_);
                child1[i] = w * parent1[i] + (1.0 - w) * parent2[i];
                child2[i] = (1.0 - w) * parent1[i] + w * parent2[i];
            }
        }
        return {child1, child2};
    }

    void updateContext(double populationDiversity, double relativeGeneration) noexcept {
        populationDiversity_ = std::clamp(populationDiversity, 0.0, 1.0);
        relativeGeneration_ = std::clamp(relativeGeneration, 0.0, 1.0);
    }

    const FuzzyAdaptiveCrossoverConfig& config() const noexcept { return config_; }

private:
    void initFuzzyEngine() {
        fis_.addInput("diversity", 0.0, 1.0);
        fis_.addInputTerm("diversity", "Low", makeTriangularMF(-0.1, 0.0, 0.5));
        fis_.addInputTerm("diversity", "High", makeTriangularMF(0.3, 1.0, 1.1));

        fis_.addInput("generation", 0.0, 1.0);
        fis_.addInputTerm("generation", "Early", makeTriangularMF(-0.1, 0.0, 0.6));
        fis_.addInputTerm("generation", "Late", makeTriangularMF(0.4, 1.0, 1.1));

        fis_.addOutput("alpha", 0.1, 0.9);
        fis_.addOutputTerm("alpha", "Small", makeTriangularMF(0.0, 0.2, 0.5));
        fis_.addOutputTerm("alpha", "Large", makeTriangularMF(0.4, 0.7, 1.0));

        fis_.addOutput("eta", 1.0, 10.0);
        fis_.addOutputTerm("eta", "Explorative", makeTriangularMF(0.0, 2.0, 5.0));
        fis_.addOutputTerm("eta", "Exploitative", makeTriangularMF(4.0, 8.0, 11.0));

        fis_.addRule("IF diversity IS Low AND generation IS Late THEN alpha IS Large");
        fis_.addRule("IF diversity IS High AND generation IS Early THEN alpha IS Small");
        fis_.addRule("IF generation IS Late THEN eta IS Exploitative");
        fis_.addRule("IF generation IS Early THEN eta IS Explorative");
    }

    FuzzyAdaptiveCrossoverConfig config_;
    MamdaniSystem fis_;
    double populationDiversity_ = 0.5;
    double relativeGeneration_ = 0.0;
    std::mt19937 rng_;
};

} // namespace fuzzy
} // namespace ga
