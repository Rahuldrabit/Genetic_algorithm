#ifndef SCRAMBLE_MUTATION_H
#define SCRAMBLE_MUTATION_H

#include "base_mutation.h"
#include <vector>

/**
 * @brief Scramble mutation operator for permutation chromosomes
 * 
 * Scramble mutation selects a random subset of the chromosome and
 * randomly reorders the elements within that subset. This is useful
 * for permutation-based problems like TSP.
 */
class ScrambleMutation : public BaseMutation {
public:
    /**
     * @brief Constructor
     * @param seed Random seed for reproducibility
     */
    explicit ScrambleMutation(unsigned int seed = std::random_device{}());

    /**
     * @brief Perform scramble mutation on a permutation
     * @param permutation The permutation chromosome to mutate
     * @param pm Mutation probability
     * @throws InvalidParameterException if pm is not in [0,1]
     */
    void mutate(std::vector<int>& permutation, double pm) const;

    /**
     * @brief Get the name of this mutation operator
     * @return String name of the operator
     */
    std::string getName() const override { return "Scramble Mutation"; }

    /**
     * @brief Get description of this mutation operator
     * @return String description
     */
    std::string getDescription() const override {
        return "Selects a random subset and randomly reorders elements within it";
    }

private:
    /**
     * @brief Perform single scramble operation
     * @param permutation The permutation to modify
     */
    void performScramble(std::vector<int>& permutation) const;
};

#endif // SCRAMBLE_MUTATION_H
