#include "simulated_binary_crossover.h"
#include <cmath>

std::pair<RealVector, RealVector> SimulatedBinaryCrossover::crossover(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        logError("Parent chromosomes have different sizes");
        return {parent1, parent2};
    }
    
    RealVector child1 = parent1;
    RealVector child2 = parent2;
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < parent1.size(); ++i) {
        if (std::abs(parent1[i] - parent2[i]) > 1e-14) {
            double u = dist(rng);
            double beta;
            
            if (u <= 0.5) {
                beta = std::pow(2.0 * u, 1.0 / (eta_c + 1.0));
            } else {
                beta = std::pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (eta_c + 1.0));
            }
            
            child1[i] = 0.5 * ((1.0 + beta) * parent1[i] + (1.0 - beta) * parent2[i]);
            child2[i] = 0.5 * ((1.0 - beta) * parent1[i] + (1.0 + beta) * parent2[i]);
        }
    }
    
    logOperation("Simulated binary crossover (SBX) with eta_c " + std::to_string(eta_c), true);
    return {child1, child2};
}
