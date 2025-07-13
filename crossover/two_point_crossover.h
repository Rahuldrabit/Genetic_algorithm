#ifndef TWO_POINT_CROSSOVER_H
#define TWO_POINT_CROSSOVER_H

#include "crossover_base.h"

class TwoPointCrossover : public CrossoverOperator {
public:
    TwoPointCrossover(unsigned seed = std::random_device{}()) : CrossoverOperator("TwoPointCrossover", seed) {}
    
    std::pair<BitString, BitString> crossover(const BitString& parent1, const BitString& parent2);
    std::pair<RealVector, RealVector> crossover(const RealVector& parent1, const RealVector& parent2);
    std::pair<IntVector, IntVector> crossover(const IntVector& parent1, const IntVector& parent2);
};

#endif // TWO_POINT_CROSSOVER_H
