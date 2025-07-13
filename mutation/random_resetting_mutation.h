#ifndef RANDOM_RESETTING_MUTATION_H
#define RANDOM_RESETTING_MUTATION_H

#include "base_mutation.h"

class RandomResettingMutation : public MutationOperator {
public:
    RandomResettingMutation(unsigned seed = std::random_device{}()) 
        : MutationOperator("RandomResettingMutation", seed) {}
    
    /**
     * @brief Random resetting mutation for integer chromosomes
     * @param chromosome Integer chromosome to mutate
     * @param pm Mutation probability per gene
     * @param minVal Minimum value for genes
     * @param maxVal Maximum value for genes
     * @throws InvalidParameterException if parameters are invalid
     */
    void mutate(std::vector<int>& chromosome, double pm, int minVal, int maxVal) const;
};

#endif // RANDOM_RESETTING_MUTATION_H
