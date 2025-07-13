#ifndef ORDER_CROSSOVER_H
#define ORDER_CROSSOVER_H

#include "crossover_base.h"

class OrderCrossover : public CrossoverOperator {
public:
    OrderCrossover(unsigned seed = std::random_device{}()) : CrossoverOperator("OrderCrossover", seed) {}
    
    std::pair<Permutation, Permutation> crossover(const Permutation& parent1, const Permutation& parent2);
};

#endif // ORDER_CROSSOVER_H
