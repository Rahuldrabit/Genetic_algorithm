#pragma once

#include "ga/core/individual.hpp"
#include "ga/core/population.hpp"

namespace ga {
namespace core {

// Abstract algorithm interface.
// Concrete implementations (GeneticAlgorithm, EvolutionStrategy, NSGA-II, etc.)
// derive from this and implement step() and population().
class IAlgorithm {
public:
    virtual ~IAlgorithm() = default;

    // Advance the algorithm by one generation.
    virtual void step() = 0;

    // Access the current population.
    virtual const Population& population() const = 0;

    // Returns the current generation count (0 before any step).
    virtual std::size_t generation() const = 0;

    // Run for a fixed number of generations.
    virtual void run(std::size_t generations) {
        for (std::size_t g = 0; g < generations; ++g) {
            step();
        }
    }
};

} // namespace core
} // namespace ga
