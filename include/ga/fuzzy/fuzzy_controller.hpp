#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace fuzzy {

struct FuzzyControllerConfig {
    // Low is zero at lowZero, medium peaks at mediumCenter, and high starts at
    // highStart. Inputs are normalized to [0, 1].
    double lowZero = 0.55;
    double mediumCenter = 0.50;
    double highStart = 0.45;
    double improvementScale = 20.0;

    // Zero-order Sugeno consequents. Every multiplier is caller-configurable.
    ga::metaheuristics::ControlSignal lowDiversityStagnant{1.55, 0.80, 1.30, 1.45};
    ga::metaheuristics::ControlSignal lowDiversitySlow{1.35, 0.90, 1.20, 1.30};
    ga::metaheuristics::ControlSignal balanced{1.00, 1.00, 1.00, 1.00};
    ga::metaheuristics::ControlSignal diverseProductive{0.78, 1.25, 0.85, 0.80};
    ga::metaheuristics::ControlSignal diverseStagnant{1.10, 0.95, 1.10, 1.15};
};

// A zero-order Sugeno-style controller. Inputs and outputs are normalized so
// one controller can adapt PSO, ACOR, GSA, and ACO without knowing their units.
class FuzzyAdaptiveController final : public ga::metaheuristics::IAdaptiveController {
public:
    explicit FuzzyAdaptiveController(FuzzyControllerConfig config = {})
        : config_(std::move(config)) {
        validate();
    }

    ga::metaheuristics::ControlSignal update(
        const ga::metaheuristics::ProgressState& state) const override {
        const auto diversity = memberships(state.normalizedDiversity);
        const auto progress = memberships(
            std::clamp(state.relativeImprovement * config_.improvementScale, 0.0, 1.0));
        const auto stagnation = memberships(state.stagnation);

        // Consequents are ordered low, medium, high. Weighted-average
        // defuzzification keeps parameter changes smooth between regimes.
        WeightedOutput exploration;
        WeightedOutput exploitation;
        WeightedOutput evaporation;
        WeightedOutput randomization;

        addSignal(exploration, exploitation, evaporation, randomization,
                  diversity[0] * stagnation[2], config_.lowDiversityStagnant);
        addSignal(exploration, exploitation, evaporation, randomization,
                  diversity[0] * progress[0], config_.lowDiversitySlow);
        addSignal(exploration, exploitation, evaporation, randomization,
                  diversity[1], config_.balanced);
        addSignal(exploration, exploitation, evaporation, randomization,
                  diversity[2] * progress[2], config_.diverseProductive);
        addSignal(exploration, exploitation, evaporation, randomization,
                  diversity[2] * stagnation[2], config_.diverseStagnant);

        ga::metaheuristics::ControlSignal signal;
        signal.exploration = exploration.value(1.0);
        signal.exploitation = exploitation.value(1.0);
        signal.evaporation = evaporation.value(1.0);
        signal.randomization = randomization.value(1.0);
        return signal;
    }

    const FuzzyControllerConfig& config() const noexcept { return config_; }

private:
    struct WeightedOutput {
        double weightedSum = 0.0;
        double weight = 0.0;

        double value(double fallback) const {
            return weight > 0.0 ? weightedSum / weight : fallback;
        }
    };

    static void addRule(WeightedOutput& output, double strength, double consequent) {
        output.weightedSum += strength * consequent;
        output.weight += strength;
    }

    static void addSignal(WeightedOutput& exploration,
                          WeightedOutput& exploitation,
                          WeightedOutput& evaporation,
                          WeightedOutput& randomization,
                          double strength,
                          const ga::metaheuristics::ControlSignal& signal) {
        addRule(exploration, strength, signal.exploration);
        addRule(exploitation, strength, signal.exploitation);
        addRule(evaporation, strength, signal.evaporation);
        addRule(randomization, strength, signal.randomization);
    }

    std::array<double, 3> memberships(double value) const {
        const double x = std::clamp(value, 0.0, 1.0);
        const double low = std::clamp(
            (config_.lowZero - x) / config_.lowZero, 0.0, 1.0);
        const double medium = x <= config_.mediumCenter
                                  ? x / config_.mediumCenter
                                  : (1.0 - x) / (1.0 - config_.mediumCenter);
        const double high = std::clamp(
            (x - config_.highStart) / (1.0 - config_.highStart), 0.0, 1.0);
        return {low, std::clamp(medium, 0.0, 1.0), high};
    }

    static bool validSignal(const ga::metaheuristics::ControlSignal& signal) {
        return std::isfinite(signal.exploration) && signal.exploration > 0.0 &&
               std::isfinite(signal.exploitation) && signal.exploitation > 0.0 &&
               std::isfinite(signal.evaporation) && signal.evaporation > 0.0 &&
               std::isfinite(signal.randomization) && signal.randomization > 0.0;
    }

    void validate() const {
        if (!std::isfinite(config_.lowZero) || !std::isfinite(config_.mediumCenter) ||
            !std::isfinite(config_.highStart) ||
            !std::isfinite(config_.improvementScale) || config_.highStart <= 0.0 ||
            config_.highStart > config_.mediumCenter ||
            config_.mediumCenter > config_.lowZero || config_.lowZero >= 1.0 ||
            config_.improvementScale <= 0.0 ||
            !validSignal(config_.lowDiversityStagnant) ||
            !validSignal(config_.lowDiversitySlow) || !validSignal(config_.balanced) ||
            !validSignal(config_.diverseProductive) ||
            !validSignal(config_.diverseStagnant)) {
            throw std::invalid_argument("invalid fuzzy-controller configuration");
        }
    }

    FuzzyControllerConfig config_;
};

} // namespace fuzzy
} // namespace ga
