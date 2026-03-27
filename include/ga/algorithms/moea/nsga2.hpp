#pragma once

#include <cstddef>
#include <functional>
#include <random>
#include <vector>
#include "ga/core/individual.hpp"

namespace ga {
namespace moea {

struct Nsga2Config {
    std::size_t populationSize = 100;
    std::size_t generations = 100;
    unsigned seed = 0;
};

struct Nsga2GenerationStats {
    std::size_t firstFrontSize = 0;
    double bestObjective0 = 0.0;
};

struct Nsga2Result {
    std::vector<ga::Individual> population;
    std::vector<Nsga2GenerationStats> history;
};

class Nsga2 {
public:
    using EvaluateFn = std::function<void(std::vector<ga::Individual>&)>;
    using ReproduceFn = std::function<std::vector<ga::Individual>(
        const std::vector<ga::Individual>&,
        const std::vector<std::size_t>&,
        std::mt19937&)>;

    explicit Nsga2(const Nsga2Config& cfg = {});

    Nsga2Result run(std::vector<ga::Individual> initialPopulation,
                    const EvaluateFn& evaluate,
                    const ReproduceFn& reproduce) const;

    std::vector<ga::Individual>
    environmentalSelect(const std::vector<ga::Individual>& combined,
                        std::size_t targetSize) const;

    // Returns fronts as vectors of population indices.
    std::vector<std::vector<std::size_t>>
    nonDominatedSort(const std::vector<ga::Individual>& population) const;

    // Returns crowding distances for the provided front, in front-local order.
    std::vector<double>
    crowdingDistance(const std::vector<ga::Individual>& population,
                     const std::vector<std::size_t>& front) const;

private:
    Nsga2Config cfg_;

    std::vector<std::size_t>
    tournamentSelect(const std::vector<ga::Individual>& population,
                     std::size_t count,
                     std::mt19937& rng) const;

    static void validateObjectives(const std::vector<ga::Individual>& population);
    static Nsga2GenerationStats summarizeGeneration(const std::vector<ga::Individual>& population,
                                                    const std::vector<std::vector<std::size_t>>& fronts);
    static bool dominates(const ga::Individual& a, const ga::Individual& b);
};

} // namespace moea
} // namespace ga
