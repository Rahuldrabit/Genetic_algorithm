#ifndef SELF_ADAPTIVE_MUTATION_H
#define SELF_ADAPTIVE_MUTATION_H

#include "base_mutation.h"
#include <vector>

/**
 * @brief Self-adaptive mutation operator with evolving step sizes
 * 
 * This mutation operator evolves both the objective variables and
 * the strategy parameters (mutation step sizes). It's commonly used
 * in Evolution Strategies (ES).
 */
class SelfAdaptiveMutation : public MutationOperator {
public:
    /**
     * @brief Structure representing an individual with self-adaptive parameters
     */
    struct SelfAdaptiveIndividual {
        std::vector<double> genes;      ///< Objective variables
        double sigma;                   ///< Strategy parameter (step size)
        
        /**
         * @brief Constructor
         * @param size Number of objective variables
         * @param initial_sigma Initial step size
         */
        SelfAdaptiveIndividual(size_t size, double initial_sigma);
    };

    /**
     * @brief Constructor
     * @param seed Random seed for reproducibility
     */
    explicit SelfAdaptiveMutation(unsigned int seed = std::random_device{}());

    /**
     * @brief Perform self-adaptive mutation on an individual
     * @param individual The individual to mutate (contains both genes and sigma)
     * @param lowerBounds Lower bounds for objective variables
     * @param upperBounds Upper bounds for objective variables
     * @param tau Learning rate for strategy parameter adaptation (default: 0.1)
     * @throws InvalidParameterException if bounds are inconsistent or tau <= 0
     */
    void mutate(SelfAdaptiveIndividual& individual,
                const std::vector<double>& lowerBounds,
                const std::vector<double>& upperBounds,
                double tau = 0.1) const;

    /**
     * @brief Get the name of this mutation operator
     * @return String name of the operator
     */
    std::string getName() const { return "Self-Adaptive Mutation"; }

    /**
     * @brief Get description of this mutation operator
     * @return String description
     */
    std::string getDescription() const {
        return "Evolves both objective variables and mutation step sizes";
    }

private:
    mutable std::normal_distribution<double> normal_dist;
    
    /**
     * @brief Validate bounds consistency
     * @param lowerBounds Lower bounds vector
     * @param upperBounds Upper bounds vector
     * @param numGenes Expected number of genes
     */
    void validateBounds(const std::vector<double>& lowerBounds,
                       const std::vector<double>& upperBounds,
                       size_t numGenes) const;
    
    /**
     * @brief Apply bounds constraints to a value
     * @param value Value to constrain
     * @param lower Lower bound
     * @param upper Upper bound
     * @return Constrained value
     */
    double applyBounds(double value, double lower, double upper) const;
};

#endif // SELF_ADAPTIVE_MUTATION_H
