#ifndef ONE_POINT_CROSSOVER_H
#define ONE_POINT_CROSSOVER_H

#include "base_crossover.h"

/**
 * @brief One-Point Crossover Operator
 * 
 * This class implements the one-point crossover operation for different
 * chromosome representations including binary, real-valued, and integer vectors.
 * 
 * The operation selects a random crossover point and exchanges the genetic
 * material between parents after that point.
 */
class OnePointCrossover : public CrossoverOperator {
public:
    /**
     * @brief Constructor
     * @param seed Random seed for reproducible results
     */
    OnePointCrossover(unsigned seed = std::random_device{}()) 
        : CrossoverOperator("OnePointCrossover", seed) {}
    
    /**
     * @brief Perform one-point crossover on binary strings
     * @param parent1 First parent chromosome
     * @param parent2 Second parent chromosome
     * @return Pair of offspring chromosomes
     */
    std::pair<BitString, BitString> crossover(const BitString& parent1, const BitString& parent2) override;
    
    /**
     * @brief Perform one-point crossover on real-valued vectors
     * @param parent1 First parent chromosome
     * @param parent2 Second parent chromosome
     * @return Pair of offspring chromosomes
     */
    std::pair<RealVector, RealVector> crossover(const RealVector& parent1, const RealVector& parent2) override;
    
    /**
     * @brief Perform one-point crossover on integer vectors
     * @param parent1 First parent chromosome
     * @param parent2 Second parent chromosome
     * @return Pair of offspring chromosomes
     */
    std::pair<IntVector, IntVector> crossover(const IntVector& parent1, const IntVector& parent2) override;
};

#endif // ONE_POINT_CROSSOVER_H
