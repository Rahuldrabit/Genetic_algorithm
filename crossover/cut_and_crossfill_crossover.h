#ifndef CUT_AND_CROSSFILL_CROSSOVER_H
#define CUT_AND_CROSSFILL_CROSSOVER_H

#include "base_crossover.h"

class CutAndCrossfillCrossover : public CrossoverOperator {
public:
    CutAndCrossfillCrossover(unsigned seed = std::random_device{}()) 
        : CrossoverOperator("CutAndCrossfillCrossover", seed) {}
    
    std::pair<Permutation, Permutation> crossover(const Permutation& parent1, const Permutation& parent2);
};

#endif // CUT_AND_CROSSFILL_CROSSOVER_H
