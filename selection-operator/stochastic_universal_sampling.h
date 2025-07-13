#ifndef STOCHASTIC_UNIVERSAL_SAMPLING_H
#define STOCHASTIC_UNIVERSAL_SAMPLING_H

#include "base_selection.h"

/**
 * @brief Stochastic Universal Sampling (SUS) Selection Operator
 * 
 * SUS is a selection method that provides better distribution than roulette wheel
 * selection by using evenly spaced selection points. This reduces bias and ensures
 * that the number of offspring for each individual is closer to its expected value.
 * 
 * The algorithm uses a single random starting point and then selects individuals
 * at regular intervals, which reduces the variance in selection compared to
 * multiple independent selections.
 */
class StochasticUniversalSampling : public SelectionOperator {
public:
    /**
     * @brief Construct a new Stochastic Universal Sampling object
     */
    StochasticUniversalSampling();
    
    /**
     * @brief Select individuals using stochastic universal sampling
     * @param population The population to select from
     * @param count Number of individuals to select
     * @return Vector of selected individuals
     */
    std::vector<Individual> select(const std::vector<Individual>& population, size_t count) override;
    
    /**
     * @brief Legacy function for backward compatibility
     * @param Population Population to select from
     * @param NumSelections Number of selections to make
     * @return Vector of selected indices
     */
    static std::vector<unsigned int> selectIndices(std::vector<Individual>& Population, 
                                                  unsigned int NumSelections);
};

// Legacy function declaration for backward compatibility
std::vector<unsigned int> StochasticUniversalSamplingLegacy(std::vector<Individual>& Population, 
                                                           unsigned int NumSelections);

#endif // STOCHASTIC_UNIVERSAL_SAMPLING_H
