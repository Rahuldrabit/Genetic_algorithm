#ifndef UNIFORM_MUTATION_H
#define UNIFORM_MUTATION_H

#include "base_mutation.h"

class UniformMutation : public MutationOperator {
public:
    UniformMutation(unsigned seed = std::random_device{}()) 
        : MutationOperator("UniformMutation", seed) {}
    
    /**
     * @brief Uniform mutation for real-valued chromosomes
     * @param chromosome Real-valued chromosome to mutate
     * @param pm Mutation probability per gene
     * @param lowerBounds Lower bounds for each gene
     * @param upperBounds Upper bounds for each gene
     * @throws InvalidParameterException if bounds are inconsistent
     */
    void mutate(std::vector<double>& chromosome, double pm,
               const std::vector<double>& lowerBounds,
               const std::vector<double>& upperBounds) const;
};

#endif // UNIFORM_MUTATION_H
