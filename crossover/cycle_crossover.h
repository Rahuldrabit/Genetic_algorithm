#ifndef CYCLE_CROSSOVER_H
#define CYCLE_CROSSOVER_H

#include "crossover_base.h"

class CycleCrossover : public CrossoverOperator {
public:
    CycleCrossover(unsigned seed = std::random_device{}()) : CrossoverOperator("CycleCrossover", seed) {}
    
    std::pair<Permutation, Permutation> crossover(const Permutation& parent1, const Permutation& parent2);
    
private:
    std::vector<std::vector<int>> findCycles(const Permutation& parent1, const Permutation& parent2);
};

#endif // CYCLE_CROSSOVER_H
