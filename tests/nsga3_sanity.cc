#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "ga/moea/nsga3.hpp"

namespace {

ga::Individual makeIndividual(double o1, double o2) {
    ga::Individual ind;
    ind.evaluation.objectives = {o1, o2};
    ind.evaluation.feasible = true;
    return ind;
}

} // namespace

int main() {
    try {
        ga::moea::Nsga3 nsga3({8, 5, 123});

        const auto refs = ga::moea::Nsga3::generateDasDennisReferencePoints(3, 4);
        if (refs.size() != 15) {
            std::cerr << "[FAIL] unexpected 3D reference point count\n";
            return 1;
        }

        std::vector<ga::Individual> pop = {
            makeIndividual(1.0, 4.0),
            makeIndividual(2.0, 3.0),
            makeIndividual(3.0, 2.0),
            makeIndividual(4.0, 1.0),
            makeIndividual(2.5, 2.5),
            makeIndividual(3.5, 3.5),
        };

        auto selected = nsga3.environmentalSelect(pop, 3, {{0.0, 1.0}, {0.5, 0.5}, {1.0, 0.0}});
        if (selected.size() != 3) {
            std::cerr << "[FAIL] NSGA-III selected wrong population size\n";
            return 1;
        }

        std::vector<ga::Individual> scaled = {
            makeIndividual(0.02, 1000.0),
            makeIndividual(0.05, 700.0),
            makeIndividual(0.20, 320.0),
            makeIndividual(0.60, 120.0),
            makeIndividual(0.95, 60.0),
            makeIndividual(1.20, 20.0),
        };
        auto scaled_selected = nsga3.environmentalSelect(scaled, 2, {{0.0, 1.0}, {1.0, 0.0}});
        if (scaled_selected.size() != 2) {
            std::cerr << "[FAIL] NSGA-III scaled selection wrong size\n";
            return 1;
        }

        bool hasMinFirst = false;
        bool hasMinSecond = false;
        for (const auto& ind : scaled_selected) {
            hasMinFirst = hasMinFirst || (ind.evaluation.objectives[0] <= 0.05);
            hasMinSecond = hasMinSecond || (ind.evaluation.objectives[1] <= 60.0);
        }
        if (!hasMinFirst || !hasMinSecond) {
            std::cerr << "[FAIL] NSGA-III scaled normalization lost extreme trade-offs\n";
            return 1;
        }

        bool hasLowFirst = false;
        bool hasLowSecond = false;
        for (const auto& ind : selected) {
            if (ind.evaluation.objectives.size() != 2) {
                std::cerr << "[FAIL] NSGA-III returned invalid objective dimension\n";
                return 1;
            }
            hasLowFirst = hasLowFirst || (ind.evaluation.objectives[0] <= 2.0);
            hasLowSecond = hasLowSecond || (ind.evaluation.objectives[1] <= 2.0);
        }

        if (!hasLowFirst || !hasLowSecond) {
            std::cerr << "[FAIL] NSGA-III niching did not preserve boundary trade-offs\n";
            return 1;
        }

        bool thrown = false;
        try {
            (void)nsga3.environmentalSelect(pop, 2, {{0.0, 1.0, 0.0}});
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        if (!thrown) {
            std::cerr << "[FAIL] NSGA-III dimensionality validation did not trigger\n";
            return 1;
        }

        std::cout << "[PASS] NSGA-III sanity checks\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] exception: " << e.what() << "\n";
        return 1;
    }
}
