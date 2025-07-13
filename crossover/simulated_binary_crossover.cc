#include "simulated_binary_crossover.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

// ============================================================================
// SIMULATED BINARY CROSSOVER (SBX) IMPLEMENTATION
// ============================================================================

std::pair<RealVector, RealVector> SimulatedBinaryCrossover::crossover(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    RealVector child1(length), child2(length);
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // For each gene position, perform SBX
    for (size_t i = 0; i < length; ++i) {
        double p1_val = parent1[i];
        double p2_val = parent2[i];
        
        // If parents are identical, children are identical
        if (std::abs(p1_val - p2_val) < 1e-14) {
            child1[i] = p1_val;
            child2[i] = p2_val;
            continue;
        }
        
        // Ensure p1_val <= p2_val
        if (p1_val > p2_val) {
            std::swap(p1_val, p2_val);
        }
        
        // Generate random number
        double u = dist(rng);
        
        // Calculate beta using the distribution
        double beta = calculateBeta(u);
        
        // Calculate offspring values
        double alpha = 2.0 - std::pow(beta, -(eta_c + 1.0));
        
        child1[i] = 0.5 * ((p1_val + p2_val) - beta * std::abs(p2_val - p1_val));
        child2[i] = 0.5 * ((p1_val + p2_val) + beta * std::abs(p2_val - p1_val));
    }
    
    return {child1, child2};
}

double SimulatedBinaryCrossover::calculateBeta(double u) const {
    double beta;
    
    if (u <= 0.5) {
        beta = std::pow(2.0 * u, 1.0 / (eta_c + 1.0));
    } else {
        beta = std::pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (eta_c + 1.0));
    }
    
    return beta;
}
