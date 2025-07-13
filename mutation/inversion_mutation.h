#ifndef INVERSION_MUTATION_H
#define INVERSION_MUTATION_H

#include "base_mutation.h"

class InversionMutation : public MutationOperator {
public:
    InversionMutation(unsigned seed = std::random_device{}()) 
        : MutationOperator("InversionMutation", seed) {}
    
    /**
     * @brief Inversion mutation for permutations
     * @param permutation Permutation to mutate
     * @param pm Mutation probability
     */
    void mutate(std::vector<int>& permutation, double pm) const;
};

#endif // INVERSION_MUTATION_H
