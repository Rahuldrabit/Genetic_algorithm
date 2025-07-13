#ifndef DIFFERENTIAL_EVOLUTION_CROSSOVER_H
#define DIFFERENTIAL_EVOLUTION_CROSSOVER_H

#include "base_crossover.h"

class DifferentialEvolutionCrossover : public CrossoverOperator {
private:
    double crossover_rate;
    
public:
    DifferentialEvolutionCrossover(double cr = 0.5, unsigned seed = std::random_device{}()) 
        : CrossoverOperator("DifferentialEvolutionCrossover", seed), crossover_rate(cr) {}
    
    RealVector performCrossover(const RealVector& target, const RealVector& mutant);
};

#endif // DIFFERENTIAL_EVOLUTION_CROSSOVER_H
