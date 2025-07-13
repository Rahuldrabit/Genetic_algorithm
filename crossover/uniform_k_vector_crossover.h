#ifndef UNIFORM_K_VECTOR_CROSSOVER_H
#define UNIFORM_K_VECTOR_CROSSOVER_H

#include "base_crossover.h"

class UniformKVectorCrossover : public CrossoverOperator {
private:
    double swap_probability;
    
public:
    UniformKVectorCrossover(double p = 0.1, unsigned seed = std::random_device{}()) 
        : CrossoverOperator("UniformKVectorCrossover", seed), swap_probability(p) {}
    
    std::vector<BitString> crossover(const std::vector<BitString>& parents);
    std::vector<RealVector> crossover(const std::vector<RealVector>& parents);
    std::vector<IntVector> crossover(const std::vector<IntVector>& parents);
};

#endif // UNIFORM_K_VECTOR_CROSSOVER_H
