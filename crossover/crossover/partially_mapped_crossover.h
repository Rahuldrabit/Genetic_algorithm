#ifndef PARTIALLY_MAPPED_CROSSOVER_H
#define PARTIALLY_MAPPED_CROSSOVER_H

#include "base_crossover.h"

class PartiallyMappedCrossover : public CrossoverOperator {
public:
    PartiallyMappedCrossover(unsigned seed = std::random_device{}()) 
        : CrossoverOperator("PartiallyMappedCrossover", seed) {}
    
    std::pair<Permutation, Permutation> crossover(const Permutation& parent1, const Permutation& parent2);
};

#endif // PARTIALLY_MAPPED_CROSSOVER_H
