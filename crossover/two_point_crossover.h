#ifndef TWO_POINT_CROSSOVER_H
#define TWO_POINT_CROSSOVER_H

#include "base_crossover.h"

/**
 * @brief Two-Point Crossover Operator
 * 
 * This class implements the two-point crossover operation for different
 * chromosome representations. Two crossover points are selected and the
 * genetic material between these points is exchanged between parents.
 */
class TwoPointCrossover : public CrossoverOperator {
public:
    /**
     * @brief Constructor
     * @param seed Random seed for reproducible results
     */
    TwoPointCrossover(unsigned seed = std::random_device{}()) 
        : CrossoverOperator("TwoPointCrossover", seed) {}
    
    /**
     * @brief Perform two-point crossover on binary strings
     * @param parent1 First parent chromosome
     * @param parent2 Second parent chromosome
     * @return Pair of offspring chromosomes
     */
    std::pair<BitString, BitString> crossover(const BitString& parent1, const BitString& parent2) override;
    
    /**
     * @brief Perform two-point crossover on real-valued vectors
     * @param parent1 First parent chromosome
     * @param parent2 Second parent chromosome
     * @return Pair of offspring chromosomes
     */
    std::pair<RealVector, RealVector> crossover(const RealVector& parent1, const RealVector& parent2) override;
    
    /**
     * @brief Perform two-point crossover on integer vectors
     * @param parent1 First parent chromosome
     * @param parent2 Second parent chromosome
     * @return Pair of offspring chromosomes
     */
    std::pair<IntVector, IntVector> crossover(const IntVector& parent1, const IntVector& parent2) override;
};

#endif // TWO_POINT_CROSSOVER_H
