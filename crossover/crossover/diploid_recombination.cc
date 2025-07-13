#include "diploid_recombination.h"

// ============================================================================
// DIPLOID RECOMBINATION IMPLEMENTATION
// ============================================================================

BitString DiploidRecombination::formGamete(const DiploidChromosome& parent) {
    operation_count++;
    
    BitString gamete;
    gamete.reserve(parent.first.size());
    
    std::uniform_int_distribution<int> dist(0, 1);
    
    for (size_t i = 0; i < parent.first.size(); ++i) {
        if (dist(rng) == 0) {
            gamete.push_back(parent.first[i]);
        } else {
            gamete.push_back(parent.second[i]);
        }
    }
    
    return gamete;
}

DiploidRecombination::DiploidChromosome DiploidRecombination::crossover(const DiploidChromosome& parent1, const DiploidChromosome& parent2) {
    BitString gamete1 = formGamete(parent1);
    BitString gamete2 = formGamete(parent2);
    
    return {gamete1, gamete2};
}
