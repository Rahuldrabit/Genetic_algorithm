#ifndef GAUSSIAN_MUTATION_H
#define GAUSSIAN_MUTATION_H

#include "base_mutation.h"

class GaussianMutation : public MutationOperator {
public:
    GaussianMutation(unsigned seed = std::random_device{}()) 
        : MutationOperator("GaussianMutation", seed) {}
    
    /**
     * @brief Gaussian perturbation mutation
     * @param chromosome Real-valued chromosome to mutate
     * @param pm Mutation probability per gene
     * @param sigma Standard deviation for Gaussian perturbation
     * @param lowerBounds Lower bounds for each gene
     * @param upperBounds Upper bounds for each gene
     * @throws InvalidParameterException if parameters are invalid
     */
    void mutate(std::vector<double>& chromosome, double pm, double sigma,
               const std::vector<double>& lowerBounds,
               const std::vector<double>& upperBounds) const;
};

#endif // GAUSSIAN_MUTATION_H
