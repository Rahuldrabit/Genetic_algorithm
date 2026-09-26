#ifndef TOURNAMENT_SELECTION_H
#define TOURNAMENT_SELECTION_H

#include "base_selection.h"

class TournamentSelection : public SelectionOperator {
private:
    size_t tournament_size;
    
public:
    TournamentSelection(size_t size = 3, unsigned seed = std::random_device{}()) 
        : SelectionOperator("Tournament", seed), tournament_size(size) {}
    
    std::vector<Individual> select(const std::vector<Individual>& population, 
                                 size_t count) override;
    
    // Standalone function for compatibility
    static std::vector<unsigned int> selectIndices(std::vector<Individual>& Population, 
                                                  unsigned int TournamentSize);
};

#endif // TOURNAMENT_SELECTION_H
