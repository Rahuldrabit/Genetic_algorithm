#pragma once

#include <memory>
#include "ga/core/evaluation.hpp"
#include "ga/core/genome.hpp"

namespace ga {

class Individual {
public:
    std::unique_ptr<IGenome> genome;
    Evaluation evaluation;
    int age = 0;

    Individual() = default;

    // Copy operations clone the genome to preserve value semantics for populations.
    Individual(const Individual& other)
        : evaluation(other.evaluation), age(other.age) {
        if (other.genome) {
            genome = other.genome->clone();
        }
    }

    Individual& operator=(const Individual& other) {
        if (this == &other) {
            return *this;
        }
        evaluation = other.evaluation;
        age = other.age;
        genome = other.genome ? other.genome->clone() : nullptr;
        return *this;
    }

    Individual(Individual&&) noexcept = default;
    Individual& operator=(Individual&&) noexcept = default;
};

} // namespace ga
