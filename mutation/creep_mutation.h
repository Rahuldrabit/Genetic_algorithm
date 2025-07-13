#ifndef CREEP_MUTATION_H
#define CREEP_MUTATION_H

#include "base_mutation.h"

class CreepMutation : public MutationOperator {
public:
    CreepMutation(unsigned seed = std::random_device{}()) 
        : MutationOperator("CreepMutation", seed) {}
    
    /**
     * @brief Creep mutation for integer chromosomes
     * @param chromosome Integer chromosome to mutate
     * @param pm Mutation probability per gene
     * @param stepSize Maximum step size for mutation
     * @param minVal Minimum value for genes
     * @param maxVal Maximum value for genes
     * @throws InvalidParameterException if parameters are invalid
     */
    void mutate(std::vector<int>& chromosome, double pm, int stepSize, int minVal, int maxVal) const;
};

#endif // CREEP_MUTATION_H
