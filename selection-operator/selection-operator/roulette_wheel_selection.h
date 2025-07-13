#ifndef ROULETTE_WHEEL_SELECTION_H
#define ROULETTE_WHEEL_SELECTION_H

#include "base_selection.h"

class RouletteWheelSelection : public SelectionOperator {
public:
    RouletteWheelSelection(unsigned seed = std::random_device{}()) 
        : SelectionOperator("RouletteWheel", seed) {}
    
    std::vector<Individual> select(const std::vector<Individual>& population, 
                                 size_t count) override;
    
    // Standalone function for compatibility
    static std::vector<unsigned int> selectIndices(std::vector<Individual>& Population, 
                                                  unsigned int NumSelections);
};

#endif // ROULETTE_WHEEL_SELECTION_H
