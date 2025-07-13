#ifndef UNIFORM_CROSSOVER_H
#define UNIFORM_CROSSOVER_H

#include "base_crossover.h"

/**
 * @brief Uniform Crossover Operator
 * 
 * This class implements uniform crossover where each gene is independently
 * chosen from either parent with a specified probability.
 */
class UniformCrossover : public CrossoverOperator {
private:
    double probability; ///< Probability of choosing gene from first parent
    
public:
    /**
     * @brief Constructor
     * @param p Probability of choosing gene from first parent (default 0.5)
     * @param seed Random seed for reproducible results
     */
    UniformCrossover(double p = 0.5, unsigned seed = std::random_device{}()) 
        : CrossoverOperator("UniformCrossover", seed), probability(p) {}
    
    /**
     * @brief Set the probability parameter
     * @param p New probability value [0.0, 1.0]
     */
    void setProbability(double p) { 
        if (p < 0.0 || p > 1.0) {
            throw std::invalid_argument("Probability must be between 0.0 and 1.0");
        }
        probability = p; 
    }
    
    /**
     * @brief Get the current probability parameter
     * @return Current probability value
     */
    double getProbability() const { return probability; }
    
    /**
     * @brief Perform uniform crossover on binary strings
     * @param parent1 First parent chromosome
     * @param parent2 Second parent chromosome
     * @return Pair of offspring chromosomes
     */
    std::pair<BitString, BitString> crossover(const BitString& parent1, const BitString& parent2) override;
    
    /**
     * @brief Perform uniform crossover on real-valued vectors
     * @param parent1 First parent chromosome
     * @param parent2 Second parent chromosome
     * @return Pair of offspring chromosomes
     */
    std::pair<RealVector, RealVector> crossover(const RealVector& parent1, const RealVector& parent2) override;
    
    /**
     * @brief Perform uniform crossover on integer vectors
     * @param parent1 First parent chromosome
     * @param parent2 Second parent chromosome
     * @return Pair of offspring chromosomes
     */
    std::pair<IntVector, IntVector> crossover(const IntVector& parent1, const IntVector& parent2) override;
};

#endif // UNIFORM_CROSSOVER_H
