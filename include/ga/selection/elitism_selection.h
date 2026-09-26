#ifndef ELITISM_SELECTION_H
#define ELITISM_SELECTION_H

#include "base_selection.h"

/**
 * @brief Elitism Selection Operator
 * 
 * Elitism selection simply selects the best individuals from the population
 * based on their fitness values. This is a deterministic selection method
 * that ensures the best individuals are always preserved.
 * 
 * This operator is commonly used in survivor selection to ensure that
 * the best individuals from one generation are carried over to the next.
 */
class ElitismSelection : public SelectionOperator {
public:
    /**
     * @brief Construct a new Elitism Selection object
     */
    ElitismSelection();
    
    /**
     * @brief Select the best individuals from the population
     * @param population The population to select from
     * @param count Number of individuals to select
     * @return Vector of selected individuals (the best ones)
     */
    std::vector<Individual> select(const std::vector<Individual>& population, size_t count) override;
    
    /**
     * @brief Legacy function for backward compatibility
     * @param Population Population to select from
     * @param NumElites Number of elite individuals to select
     * @return Vector of selected indices
     */
    static std::vector<unsigned int> selectIndices(std::vector<Individual>& Population, 
                                                  unsigned int NumElites);
};

// Legacy function declaration for backward compatibility
std::vector<unsigned int> ElitismSelectionLegacy(std::vector<Individual>& Population, 
                                                unsigned int NumElites);

#endif // ELITISM_SELECTION_H
