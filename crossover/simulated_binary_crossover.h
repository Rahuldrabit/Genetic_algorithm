#ifndef SIMULATED_BINARY_CROSSOVER_H
#define SIMULATED_BINARY_CROSSOVER_H

#include "crossover_base.h"

class SimulatedBinaryCrossover : public CrossoverOperator {
private:
    double eta_c;  // distribution index
    
public:
    SimulatedBinaryCrossover(double eta = 2.0, unsigned seed = std::random_device{}()) 
        : CrossoverOperator("SimulatedBinaryCrossover", seed), eta_c(eta) {}
    
    std::pair<RealVector, RealVector> crossover(const RealVector& parent1, const RealVector& parent2);
};

#endif // SIMULATED_BINARY_CROSSOVER_H
