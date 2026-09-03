#pragma once

#include <algorithm>
#include <array>

#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace fuzzy {

// A zero-order Sugeno-style controller. Inputs and outputs are normalized so
// one controller can adapt PSO, ACOR, GSA, and ACO without knowing their units.
class FuzzyAdaptiveController final : public ga::metaheuristics::IAdaptiveController {
public:
    ga::metaheuristics::ControlSignal update(
        const ga::metaheuristics::ProgressState& state) const override {
        const auto diversity = memberships(state.normalizedDiversity);
        const auto progress = memberships(
            std::clamp(state.relativeImprovement * 20.0, 0.0, 1.0));
        const auto stagnation = memberships(state.stagnation);

        // Consequents are ordered low, medium, high. Weighted-average
        // defuzzification keeps parameter changes smooth between regimes.
        WeightedOutput exploration;
        WeightedOutput exploitation;
        WeightedOutput evaporation;
        WeightedOutput randomization;

        addRule(exploration, diversity[0] * stagnation[2], 1.55);
        addRule(exploitation, diversity[0] * stagnation[2], 0.80);
        addRule(evaporation, diversity[0] * stagnation[2], 1.30);
        addRule(randomization, diversity[0] * stagnation[2], 1.45);

        addRule(exploration, diversity[0] * progress[0], 1.35);
        addRule(exploitation, diversity[0] * progress[0], 0.90);
        addRule(evaporation, diversity[0] * progress[0], 1.20);
        addRule(randomization, diversity[0] * progress[0], 1.30);

        addRule(exploration, diversity[1], 1.00);
        addRule(exploitation, diversity[1], 1.00);
        addRule(evaporation, diversity[1], 1.00);
        addRule(randomization, diversity[1], 1.00);

        addRule(exploration, diversity[2] * progress[2], 0.78);
        addRule(exploitation, diversity[2] * progress[2], 1.25);
        addRule(evaporation, diversity[2] * progress[2], 0.85);
        addRule(randomization, diversity[2] * progress[2], 0.80);

        addRule(exploration, diversity[2] * stagnation[2], 1.10);
        addRule(exploitation, diversity[2] * stagnation[2], 0.95);
        addRule(evaporation, diversity[2] * stagnation[2], 1.10);
        addRule(randomization, diversity[2] * stagnation[2], 1.15);

        ga::metaheuristics::ControlSignal signal;
        signal.exploration = exploration.value(1.0);
        signal.exploitation = exploitation.value(1.0);
        signal.evaporation = evaporation.value(1.0);
        signal.randomization = randomization.value(1.0);
        return signal;
    }

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

    static std::array<double, 3> memberships(double value) {
        const double x = std::clamp(value, 0.0, 1.0);
        const double low = std::clamp((0.55 - x) / 0.55, 0.0, 1.0);
        const double medium = std::clamp(1.0 - std::abs(x - 0.5) / 0.5, 0.0, 1.0);
        const double high = std::clamp((x - 0.45) / 0.55, 0.0, 1.0);
        return {low, medium, high};
    }
};

} // namespace fuzzy
} // namespace ga
