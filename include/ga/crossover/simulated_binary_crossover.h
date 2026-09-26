#ifndef SIMULATED_BINARY_CROSSOVER_H
#define SIMULATED_BINARY_CROSSOVER_H

#include "base_crossover.h"

/**
 * @brief Simulated Binary Crossover (SBX) Operator
 * 
 * This class implements the Simulated Binary Crossover (SBX) operator
 * for real-valued chromosomes. SBX mimics the behavior of single-point
 * crossover on binary strings but works directly on real values.
 */
class SimulatedBinaryCrossover : public CrossoverOperator {
private:
    double eta_c; ///< Distribution index (controls spread of offspring)
    
public:
    /**
     * @brief Constructor
     * @param eta Distribution index parameter (default 2.0)
     * @param seed Random seed for reproducible results
     */
    SimulatedBinaryCrossover(double eta = 2.0, unsigned seed = std::random_device{}()) 
        : CrossoverOperator("SimulatedBinaryCrossover", seed), eta_c(eta) {}
    
    /**
     * @brief Set the distribution index parameter
     * @param eta New eta value (should be > 0.0)
     */
    void setEta(double eta) { 
        if (eta <= 0.0) {
            throw std::invalid_argument("Eta must be positive");
        }
        eta_c = eta; 
    }
    
    /**
     * @brief Get the current distribution index parameter
     * @return Current eta value
     */
    double getEta() const { return eta_c; }
    
    /**
     * @brief Perform SBX crossover on real-valued vectors
     * @param parent1 First parent chromosome
     * @param parent2 Second parent chromosome
     * @return Pair of offspring chromosomes
     */
    std::pair<RealVector, RealVector> crossover(const RealVector& parent1, const RealVector& parent2) override;
    
    // Note: SBX is specifically designed for real-valued representations
    std::pair<BitString, BitString> crossover(const BitString& /* parent1 */, const BitString& /* parent2 */) override {
        throw std::runtime_error("SBX crossover is not applicable to binary strings");
    }
    
    std::pair<IntVector, IntVector> crossover(const IntVector& /* parent1 */, const IntVector& /* parent2 */) override {
        throw std::runtime_error("SBX crossover is not applicable to integer vectors");
    }

private:
    /**
     * @brief Calculate the beta value for SBX
     * @param u Random number [0,1]
     * @return Beta value for crossover calculation
     */
    double calculateBeta(double u) const;
};

#endif // SIMULATED_BINARY_CROSSOVER_H
