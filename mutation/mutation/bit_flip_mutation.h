#ifndef BIT_FLIP_MUTATION_H
#define BIT_FLIP_MUTATION_H

#include "base_mutation.h"

class BitFlipMutation : public MutationOperator {
public:
    BitFlipMutation(unsigned seed = std::random_device{}()) 
        : MutationOperator("BitFlipMutation", seed) {}
    
    /**
     * @brief Bit-flip mutation for binary chromosomes
     * @param chromosome Binary chromosome to mutate
     * @param pm Mutation probability per bit
     * @throws InvalidParameterException if pm is not in [0,1]
     */
    void mutate(std::vector<bool>& chromosome, double pm) const;
    
    /**
     * @brief Bit-flip mutation for binary string representation
     * @param binaryString Binary string to mutate
     * @param pm Mutation probability per bit
     * @throws InvalidParameterException if pm is not in [0,1]
     */
    void mutate(std::string& binaryString, double pm) const;
};

#endif // BIT_FLIP_MUTATION_H
