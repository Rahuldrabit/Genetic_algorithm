#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "ga/algorithms/moea/nsga2.hpp"

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
        ga::moea::Nsga2 nsga2({6, 3, 123});

        std::vector<ga::Individual> pop = {
            makeIndividual(1.0, 4.0),
            makeIndividual(2.0, 3.0),
            makeIndividual(3.0, 2.0),
            makeIndividual(4.0, 1.0),
            makeIndividual(3.0, 4.0),
            makeIndividual(4.0, 3.0),
        };

        auto fronts = nsga2.nonDominatedSort(pop);
        if (fronts.empty() || fronts[0].size() != 4) {
            std::cerr << "[FAIL] unexpected first front size\n";
            return 1;
        }

        auto cd = nsga2.crowdingDistance(pop, fronts[0]);
        if (cd.size() != fronts[0].size()) {
            std::cerr << "[FAIL] crowding distance size mismatch\n";
            return 1;
        }

        std::size_t inf_count = 0;
        for (double d : cd) {
            if (std::isinf(d)) {
                ++inf_count;
            }
        }
        if (inf_count < 2) {
            std::cerr << "[FAIL] expected at least two boundary individuals with infinite crowding\n";
            return 1;
        }

        auto evaluate = [](std::vector<ga::Individual>& individuals) {
            for (auto& ind : individuals) {
                if (ind.evaluation.objectives.empty()) {
                    throw std::runtime_error("test evaluate expects prefilled objectives");
                }
            }
        };

        auto reproduce = [](const std::vector<ga::Individual>& parents,
                            const std::vector<std::size_t>& mating,
                            std::mt19937& rng) {
            std::normal_distribution<double> noise(0.0, 0.1);
            std::vector<ga::Individual> offspring;
            offspring.reserve(mating.size());
            for (std::size_t idx : mating) {
                ga::Individual child = parents[idx];
                child.evaluation.objectives[0] = std::max(0.0, child.evaluation.objectives[0] + noise(rng));
                child.evaluation.objectives[1] = std::max(0.0, child.evaluation.objectives[1] + noise(rng));
                offspring.push_back(std::move(child));
            }
            return offspring;
        };

        auto result = nsga2.run(pop, evaluate, reproduce);
        if (result.population.size() != 6) {
            std::cerr << "[FAIL] unexpected final population size\n";
            return 1;
        }
        if (result.history.size() != 4) {
            std::cerr << "[FAIL] expected history size of generations+1\n";
            return 1;
        }
        if (!std::isfinite(result.history.back().bestObjective0)) {
            std::cerr << "[FAIL] best objective summary is not finite\n";
            return 1;
        }

        std::cout << "[PASS] NSGA-II sanity checks\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] exception: " << e.what() << "\n";
        return 1;
    }
}
