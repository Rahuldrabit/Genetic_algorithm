#ifndef INSERT_MUTATION_H
#define INSERT_MUTATION_H

#include "base_mutation.h"
#include <vector>

/**
 * @brief Insert mutation operator for permutation chromosomes
 * 
 * Insert mutation selects a random element and inserts it at a random
 * position in the permutation. This is useful for permutation-based
 * problems like TSP.
 */
class InsertMutation : public MutationOperator {
public:
    /**
     * @brief Constructor
     * @param seed Random seed for reproducibility
     */
    explicit InsertMutation(unsigned int seed = std::random_device{}());

    /**
     * @brief Perform insert mutation on a permutation
     * @param permutation The permutation chromosome to mutate
     * @param pm Mutation probability
     * @throws InvalidParameterException if pm is not in [0,1]
     */
    void mutate(std::vector<int>& permutation, double pm) const;

    /**
     * @brief Get the name of this mutation operator
     * @return String name of the operator
     */
    std::string getName() const { return "Insert Mutation"; }

    /**
     * @brief Get description of this mutation operator
     * @return String description
     */
    std::string getDescription() const {
        return "Selects a random element and inserts it at a random position";
    }

private:
    /**
     * @brief Perform single insert operation
     * @param permutation The permutation to modify
     */
    void performInsert(std::vector<int>& permutation) const;
};

#endif // INSERT_MUTATION_H
