#ifndef BLEND_CROSSOVER_H
#define BLEND_CROSSOVER_H

#include "crossover_base.h"

class BlendCrossover : public CrossoverOperator {
private:
    double alpha;
    
public:
    BlendCrossover(double a = 0.5, unsigned seed = std::random_device{}()) 
        : CrossoverOperator("BlendCrossover", seed), alpha(a) {}
    
    std::pair<RealVector, RealVector> crossover(const RealVector& parent1, const RealVector& parent2);
};

#endif // BLEND_CROSSOVER_H
