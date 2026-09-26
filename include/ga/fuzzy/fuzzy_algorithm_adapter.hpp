#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "ga/fuzzy/mamdani_system.hpp"
#include "ga/fuzzy/sugeno_system.hpp"
#include "ga/fuzzy/fcm_swarm_analyzer.hpp"
#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace fuzzy {

enum class FuzzyControllerBackend { Mamdani, Sugeno };

struct FuzzyAlgorithmAdapterConfig {
    FuzzyControllerBackend backend = FuzzyControllerBackend::Mamdani;
    double improvementScale = 20.0;
    bool enableFcm = false;
    std::size_t fcmClusters = 3;
};

class FuzzyAlgorithmAdapter final : public ga::metaheuristics::IAdaptiveController {
public:
    explicit FuzzyAlgorithmAdapter(FuzzyAlgorithmAdapterConfig config = {})
        : config_(config) {
        if (config_.backend == FuzzyControllerBackend::Mamdani) {
            initDefaultMamdani();
        } else {
            initDefaultSugeno();
        }
    }

    explicit FuzzyAlgorithmAdapter(MamdaniSystem mamdani)
        : mamdani_(std::move(mamdani)) {
        config_.backend = FuzzyControllerBackend::Mamdani;
    }

    explicit FuzzyAlgorithmAdapter(SugenoSystem sugeno)
        : sugeno_(std::move(sugeno)) {
        config_.backend = FuzzyControllerBackend::Sugeno;
    }

    ga::metaheuristics::ControlSignal update(
        const ga::metaheuristics::ProgressState& state) const override {
        const double div = std::clamp(state.normalizedDiversity, 0.0, 1.0);
        const double stag = std::clamp(state.stagnation, 0.0, 1.0);
        const double prog = std::clamp(state.relativeImprovement * config_.improvementScale, 0.0, 1.0);

        std::map<std::string, double> in = {{"diversity", div}, {"stagnation", stag}, {"progress", prog}};
        ga::metaheuristics::ControlSignal signal;

        if (config_.backend == FuzzyControllerBackend::Mamdani) {
            signal.exploration = mamdani_.evaluateSingle("exploration", in, 1.0);
            signal.exploitation = mamdani_.evaluateSingle("exploitation", in, 1.0);
            signal.randomization = mamdani_.evaluateSingle("randomization", in, 1.0);
            lastShakeIntensity_ = mamdani_.evaluateSingle("shakeIntensity", in, 0.0);
        } else {
            signal.exploration = sugeno_.evaluateSingle("exploration", in, 1.0);
            signal.exploitation = sugeno_.evaluateSingle("exploitation", in, 1.0);
            signal.randomization = sugeno_.evaluateSingle("randomization", in, 1.0);
            lastShakeIntensity_ = sugeno_.evaluateSingle("shakeIntensity", in, 0.0);
        }
        return signal;
    }

    ga::metaheuristics::ControlSignal updateWithFcm(
        const ga::metaheuristics::ProgressState& state,
        const FcmSwarmMetrics& fcmMetrics) const {
        auto signal = update(state);
        if (fcmMetrics.dominantCrowding > 0.6) {
            signal.exploration *= 1.25;
            signal.randomization *= 1.30;
            lastShakeIntensity_ = std::max(lastShakeIntensity_, 0.5);
        }
        return signal;
    }

    double lastShakeIntensity() const noexcept { return lastShakeIntensity_; }
    const MamdaniSystem& mamdani() const noexcept { return mamdani_; }
    const SugenoSystem& sugeno() const noexcept { return sugeno_; }
    const FuzzyAlgorithmAdapterConfig& config() const noexcept { return config_; }

private:
    void initDefaultMamdani() {
        mamdani_.addInput("diversity", 0.0, 1.0);
        mamdani_.addInputTerm("diversity", "Low", makeTriangularMF(-0.1, 0.0, 0.5));
        mamdani_.addInputTerm("diversity", "Medium", makeTriangularMF(0.2, 0.5, 0.8));
        mamdani_.addInputTerm("diversity", "High", makeTriangularMF(0.5, 1.0, 1.1));

        mamdani_.addInput("stagnation", 0.0, 1.0);
        mamdani_.addInputTerm("stagnation", "Low", makeTriangularMF(-0.1, 0.0, 0.5));
        mamdani_.addInputTerm("stagnation", "High", makeTriangularMF(0.4, 1.0, 1.1));

        mamdani_.addInput("progress", 0.0, 1.0);
        mamdani_.addInputTerm("progress", "Low", makeTriangularMF(-0.1, 0.0, 0.5));
        mamdani_.addInputTerm("progress", "High", makeTriangularMF(0.4, 1.0, 1.1));

        mamdani_.addOutput("exploration", 0.5, 2.0);
        mamdani_.addOutputTerm("exploration", "Low", makeTriangularMF(0.4, 0.7, 1.0));
        mamdani_.addOutputTerm("exploration", "Medium", makeTriangularMF(0.8, 1.0, 1.3));
        mamdani_.addOutputTerm("exploration", "High", makeTriangularMF(1.1, 1.6, 2.1));

        mamdani_.addOutput("exploitation", 0.5, 2.0);
        mamdani_.addOutputTerm("exploitation", "Low", makeTriangularMF(0.4, 0.7, 1.0));
        mamdani_.addOutputTerm("exploitation", "Medium", makeTriangularMF(0.8, 1.0, 1.3));
        mamdani_.addOutputTerm("exploitation", "High", makeTriangularMF(1.1, 1.6, 2.1));

        mamdani_.addOutput("randomization", 0.5, 2.0);
        mamdani_.addOutputTerm("randomization", "Low", makeTriangularMF(0.4, 0.8, 1.0));
        mamdani_.addOutputTerm("randomization", "High", makeTriangularMF(1.0, 1.5, 2.1));

        mamdani_.addOutput("shakeIntensity", 0.0, 1.0);
        mamdani_.addOutputTerm("shakeIntensity", "None", makeTriangularMF(-0.1, 0.0, 0.3));
        mamdani_.addOutputTerm("shakeIntensity", "Moderate", makeTriangularMF(0.2, 0.5, 0.8));
        mamdani_.addOutputTerm("shakeIntensity", "Strong", makeTriangularMF(0.6, 1.0, 1.1));

        mamdani_.addRule("IF diversity IS Low AND stagnation IS High THEN exploration IS High");
        mamdani_.addRule("IF diversity IS Low AND stagnation IS High THEN shakeIntensity IS Strong");
        mamdani_.addRule("IF diversity IS High AND progress IS High THEN exploitation IS High");
        mamdani_.addRule("IF diversity IS Medium THEN exploration IS Medium");
        mamdani_.addRule("IF diversity IS Medium THEN exploitation IS Medium");
    }

    void initDefaultSugeno() {
        sugeno_.addInput("diversity", 0.0, 1.0);
        sugeno_.addInputTerm("diversity", "Low", makeTriangularMF(-0.1, 0.0, 0.5));
        sugeno_.addInputTerm("diversity", "Medium", makeTriangularMF(0.2, 0.5, 0.8));
        sugeno_.addInputTerm("diversity", "High", makeTriangularMF(0.5, 1.0, 1.1));

        sugeno_.addInput("stagnation", 0.0, 1.0);
        sugeno_.addInputTerm("stagnation", "Low", makeTriangularMF(-0.1, 0.0, 0.5));
        sugeno_.addInputTerm("stagnation", "High", makeTriangularMF(0.4, 1.0, 1.1));

        sugeno_.addOutput("exploration", 1.0);
        sugeno_.addOutput("exploitation", 1.0);
        sugeno_.addOutput("randomization", 1.0);
        sugeno_.addOutput("shakeIntensity", 0.0);

        SugenoRule r1({{"diversity", "Low"}, {"stagnation", "High"}}, {{"exploration", 1.5, {}}, {"shakeIntensity", 0.8, {}}});
        SugenoRule r2({{"diversity", "High"}, {"stagnation", "Low"}}, {{"exploitation", 1.4, {}}, {"shakeIntensity", 0.0, {}}});
        SugenoRule r3({{"diversity", "Medium"}}, {{"exploration", 1.0, {}}, {"exploitation", 1.0, {}}});
        sugeno_.addRule(std::move(r1));
        sugeno_.addRule(std::move(r2));
        sugeno_.addRule(std::move(r3));
    }

    FuzzyAlgorithmAdapterConfig config_;
    MamdaniSystem mamdani_;
    SugenoSystem sugeno_;
    mutable double lastShakeIntensity_ = 0.0;
};

} // namespace fuzzy
} // namespace ga
