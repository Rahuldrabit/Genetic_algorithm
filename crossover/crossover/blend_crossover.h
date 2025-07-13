#ifndef BLEND_CROSSOVER_H
#define BLEND_CROSSOVER_H

#include "base_crossover.h"

/**
 * @brief Blend Crossover (BLX-α) Operator
 * 
 * This class implements the blend crossover (BLX-α) specifically designed
 * for real-valued chromosomes. It creates offspring by blending parent
 * values with some additional exploration around the parent interval.
 */
class BlendCrossover : public CrossoverOperator {
private:
    double alpha; ///< Extension factor for the interval
    
public:
    /**
     * @brief Constructor
     * @param a Alpha parameter for interval extension (default 0.5)
     * @param seed Random seed for reproducible results
     */
    BlendCrossover(double a = 0.5, unsigned seed = std::random_device{}()) 
        : CrossoverOperator("BlendCrossover", seed), alpha(a) {}
    
    /**
     * @brief Set the alpha parameter
     * @param a New alpha value (should be >= 0.0)
     */
    void setAlpha(double a) { 
        if (a < 0.0) {
            throw std::invalid_argument("Alpha must be non-negative");
        }
        alpha = a; 
    }
    
    /**
     * @brief Get the current alpha parameter
     * @return Current alpha value
     */
    double getAlpha() const { return alpha; }
    
    /**
     * @brief Perform blend crossover on real-valued vectors
     * @param parent1 First parent chromosome
     * @param parent2 Second parent chromosome
     * @return Pair of offspring chromosomes
     */
    std::pair<RealVector, RealVector> crossover(const RealVector& parent1, const RealVector& parent2) override;
    
    // Note: BLX-α is specifically designed for real-valued representations
    // Binary and integer versions are not implemented as they don't make sense
    std::pair<BitString, BitString> crossover(const BitString& /* parent1 */, const BitString& /* parent2 */) override {
        throw std::runtime_error("BLX-α crossover is not applicable to binary strings");
    }
    
    std::pair<IntVector, IntVector> crossover(const IntVector& /* parent1 */, const IntVector& /* parent2 */) override {
        throw std::runtime_error("BLX-α crossover is not applicable to integer vectors");
    }
};

#endif // BLEND_CROSSOVER_H
