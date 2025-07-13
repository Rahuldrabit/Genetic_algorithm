#ifndef INTERMEDIATE_RECOMBINATION_H
#define INTERMEDIATE_RECOMBINATION_H

#include "crossover_base.h"

class IntermediateRecombination : public CrossoverOperator {
private:
    double alpha;
    
public:
    IntermediateRecombination(double a = 0.5, unsigned seed = std::random_device{}()) 
        : CrossoverOperator("IntermediateRecombination", seed), alpha(a) {}
    
    std::pair<RealVector, RealVector> crossover(const RealVector& parent1, const RealVector& parent2);
    RealVector singleArithmeticRecombination(const RealVector& parent1, const RealVector& parent2);
    RealVector wholeArithmeticRecombination(const RealVector& parent1, const RealVector& parent2);
};

#endif // INTERMEDIATE_RECOMBINATION_H
