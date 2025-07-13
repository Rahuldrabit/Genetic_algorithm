#ifndef LINE_RECOMBINATION_H
#define LINE_RECOMBINATION_H

#include "base_crossover.h"

class LineRecombination : public CrossoverOperator {
private:
    double extension_factor;
    
public:
    LineRecombination(double p = 0.1, unsigned seed = std::random_device{}()) 
        : CrossoverOperator("LineRecombination", seed), extension_factor(p) {}
    
    std::pair<RealVector, RealVector> crossover(const RealVector& parent1, const RealVector& parent2);
};

#endif // LINE_RECOMBINATION_H
