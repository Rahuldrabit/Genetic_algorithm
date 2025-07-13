#ifndef MULTI_POINT_CROSSOVER_H
#define MULTI_POINT_CROSSOVER_H

#include "base_crossover.h"

/**
 * @brief Multi-Point Crossover Operator
 * 
 * This class implements the multi-point crossover operation where
 * multiple crossover points are selected and genetic material is
 * exchanged at alternating segments.
 */
class MultiPointCrossover : public CrossoverOperator {
private:
    int num_points; ///< Number of crossover points
    
public:
    /**
     * @brief Constructor
     * @param points Number of crossover points (default 3)
     * @param seed Random seed for reproducible results
     */
    MultiPointCrossover(int points = 3, unsigned seed = std::random_device{}()) 
        : CrossoverOperator("MultiPointCrossover", seed), num_points(points) {
        if (points < 1) {
            throw std::invalid_argument("Number of points must be at least 1");
        }
    }
    
    /**
     * @brief Set the number of crossover points
     * @param points New number of points (must be >= 1)
     */
    void setNumPoints(int points) { 
        if (points < 1) {
            throw std::invalid_argument("Number of points must be at least 1");
        }
        num_points = points; 
    }
    
    /**
     * @brief Get the current number of crossover points
     * @return Current number of points
     */
    int getNumPoints() const { return num_points; }
    
    /**
     * @brief Perform multi-point crossover on binary strings
     * @param parent1 First parent chromosome
     * @param parent2 Second parent chromosome
     * @return Pair of offspring chromosomes
     */
    std::pair<BitString, BitString> crossover(const BitString& parent1, const BitString& parent2) override;
    
    /**
     * @brief Perform multi-point crossover on real-valued vectors
     * @param parent1 First parent chromosome
     * @param parent2 Second parent chromosome
     * @return Pair of offspring chromosomes
     */
    std::pair<RealVector, RealVector> crossover(const RealVector& parent1, const RealVector& parent2) override;
    
    /**
     * @brief Perform multi-point crossover on integer vectors
     * @param parent1 First parent chromosome
     * @param parent2 Second parent chromosome
     * @return Pair of offspring chromosomes
     */
    std::pair<IntVector, IntVector> crossover(const IntVector& parent1, const IntVector& parent2) override;

private:
    /**
     * @brief Generate sorted crossover points
     * @param length Length of the chromosome
     * @return Vector of sorted crossover points
     */
    std::vector<size_t> generateCrossoverPoints(size_t length);
};

#endif // MULTI_POINT_CROSSOVER_H
