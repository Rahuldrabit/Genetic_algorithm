#ifndef SWAP_MUTATION_H
#define SWAP_MUTATION_H

#include "base_mutation.h"

class SwapMutation : public MutationOperator {
public:
    SwapMutation(unsigned seed = std::random_device{}()) 
        : MutationOperator("SwapMutation", seed) {}
    
    /**
     * @brief Swap mutation for permutations
     * @param permutation Permutation to mutate
     * @param pm Mutation probability
     */
    void mutate(std::vector<int>& permutation, double pm) const;
};

#endif // SWAP_MUTATION_H
