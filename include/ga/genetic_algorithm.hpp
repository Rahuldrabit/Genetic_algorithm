#pragma once

#include <memory>
#include <random>
#include <utility>
#include <vector>
#include "ga/config.hpp"

// Forward declare operator base types from existing code
class MutationOperator;
class CrossoverOperator;

namespace ga {

class GeneticAlgorithm {
public:
    explicit GeneticAlgorithm(const Config& cfg);
    ~GeneticAlgorithm();

    // Run GA for provided fitness, return statistics
    Result run(const Fitness& fitness);

    // Access to operators for customization
    void setMutationOperator(std::unique_ptr<MutationOperator> op);
    void setCrossoverOperator(std::unique_ptr<CrossoverOperator> op);

    const Config& config() const { return cfg_; }

private:
    struct Individual {
        std::vector<double> genes;
        double fitness = 0.0;
    };

    Config cfg_;
    std::mt19937 rng_;

    std::unique_ptr<MutationOperator> mutation_;
    std::unique_ptr<CrossoverOperator> crossover_;

    std::vector<Individual> initPopulation_(const Fitness& f);
    std::pair<Individual, Individual> crossoverPair_(const Individual& p1, const Individual& p2, const Fitness& f);
    void mutate_(Individual& ind);
};

// Convenience factories (implemented in .cpp using existing operators)
std::unique_ptr<MutationOperator> makeGaussianMutation(unsigned seed = 0);
std::unique_ptr<MutationOperator> makeUniformMutation(unsigned seed = 0);
std::unique_ptr<CrossoverOperator> makeOnePointCrossover(unsigned seed = 0);
std::unique_ptr<CrossoverOperator> makeTwoPointCrossover(unsigned seed = 0);

} // namespace ga
