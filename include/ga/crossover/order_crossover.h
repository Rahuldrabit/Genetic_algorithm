#ifndef ORDER_CROSSOVER_H
#define ORDER_CROSSOVER_H

#include "base_crossover.h"

/**
 * @brief Order Crossover (OX) Operator
 * 
 * This class implements the Order Crossover (OX) operator specifically
 * designed for permutation representations. It preserves the relative
 * order of elements from one parent while filling in elements from
 * the other parent.
 */
class OrderCrossover : public CrossoverOperator {
public:
    /**
     * @brief Constructor
     * @param seed Random seed for reproducible results
     */
    OrderCrossover(unsigned seed = std::random_device{}()) 
        : CrossoverOperator("OrderCrossover", seed) {}
    
    /**
     * @brief Perform order crossover on permutation
     * @param parent1 First parent permutation
     * @param parent2 Second parent permutation
     * @return Pair of offspring permutations
     */
    std::pair<IntVector, IntVector> crossover(const IntVector& parent1, const IntVector& parent2) override;
    
    // Order crossover is specifically for permutations
    std::pair<BitString, BitString> crossover(const BitString& /* parent1 */, const BitString& /* parent2 */) override {
        throw std::runtime_error("Order crossover is only applicable to permutations");
    }
    
    std::pair<RealVector, RealVector> crossover(const RealVector& /* parent1 */, const RealVector& /* parent2 */) override {
        throw std::runtime_error("Order crossover is only applicable to permutations");
    }

private:
    /**
     * @brief Create one offspring using order crossover
     * @param p1 First parent
     * @param p2 Second parent
     * @return Single offspring
     */
    IntVector createOffspring(const IntVector& p1, const IntVector& p2);
};

#endif // ORDER_CROSSOVER_H
