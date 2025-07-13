#ifndef UNIFORM_CROSSOVER_H
#define UNIFORM_CROSSOVER_H

#include "crossover_base.h"

class UniformCrossover : public CrossoverOperator {
private:
    double probability;
    
public:
    UniformCrossover(double p = 0.5, unsigned seed = std::random_device{}()) 
        : CrossoverOperator("UniformCrossover", seed), probability(p) {}
    
    std::pair<BitString, BitString> crossover(const BitString& parent1, const BitString& parent2);
    std::pair<RealVector, RealVector> crossover(const RealVector& parent1, const RealVector& parent2);
    std::pair<IntVector, IntVector> crossover(const IntVector& parent1, const IntVector& parent2);
};

#endif // UNIFORM_CROSSOVER_H
