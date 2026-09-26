#ifndef LIST_MUTATION_H
#define LIST_MUTATION_H

#include "base_mutation.h"
#include <vector>

/**
 * @brief List mutation operator for variable-length chromosomes
 * 
 * This mutation operator can modify both the content and size of
 * variable-length list chromosomes. It's useful for problems where
 * the solution length is not fixed.
 */
class ListMutation : public MutationOperator {
public:
    /**
     * @brief Constructor
     * @param seed Random seed for reproducibility
     */
    explicit ListMutation(unsigned int seed = std::random_device{}());

    /**
     * @brief Perform list mutation on a variable-length chromosome
     * @param list The list chromosome to mutate
     * @param pm_content Mutation probability for content changes
     * @param pm_size Mutation probability for size changes
     * @param minVal Minimum value for list elements
     * @param maxVal Maximum value for list elements
     * @param minSize Minimum allowed list size
     * @param maxSize Maximum allowed list size
     * @throws InvalidParameterException if parameters are invalid
     */
    void mutate(std::vector<int>& list, 
                double pm_content, 
                double pm_size,
                int minVal, 
                int maxVal, 
                size_t minSize, 
                size_t maxSize) const;

    /**
     * @brief Get the name of this mutation operator
     * @return String name of the operator
     */
    std::string getName() const { return "List Mutation"; }

    /**
     * @brief Get description of this mutation operator
     * @return String description
     */
    std::string getDescription() const {
        return "Mutates both content and size of variable-length lists";
    }

private:
    /**
     * @brief Mutate list content (change existing values)
     * @param list The list to mutate
     * @param pm_content Content mutation probability
     * @param minVal Minimum value for elements
     * @param maxVal Maximum value for elements
     */
    void mutateContent(std::vector<int>& list, 
                      double pm_content,
                      int minVal, 
                      int maxVal) const;
    
    /**
     * @brief Mutate list size (add or remove elements)
     * @param list The list to mutate
     * @param pm_size Size mutation probability
     * @param minVal Minimum value for new elements
     * @param maxVal Maximum value for new elements
     * @param minSize Minimum allowed list size
     * @param maxSize Maximum allowed list size
     */
    void mutateSize(std::vector<int>& list,
                   double pm_size,
                   int minVal,
                   int maxVal,
                   size_t minSize,
                   size_t maxSize) const;
    
    /**
     * @brief Validate parameters
     * @param pm_content Content mutation probability
     * @param pm_size Size mutation probability
     * @param minVal Minimum value
     * @param maxVal Maximum value
     * @param minSize Minimum size
     * @param maxSize Maximum size
     */
    void validateParameters(double pm_content, 
                           double pm_size,
                           int minVal, 
                           int maxVal,
                           size_t minSize, 
                           size_t maxSize) const;
};

#endif // LIST_MUTATION_H
