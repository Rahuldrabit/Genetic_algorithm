#ifndef EDGE_CROSSOVER_H
#define EDGE_CROSSOVER_H

#include "base_crossover.h"
#include <map>
#include <set>

class EdgeCrossover : public CrossoverOperator {
public:
    EdgeCrossover(unsigned seed = std::random_device{}()) 
        : CrossoverOperator("EdgeCrossover", seed) {}
    
    Permutation performCrossover(const Permutation& parent1, const Permutation& parent2);
    
private:
    std::map<int, std::set<int>> buildEdgeTable(const Permutation& parent1, const Permutation& parent2);
};

#endif // EDGE_CROSSOVER_H
