#include "line_recombination.h"
#include <stdexcept>

// ============================================================================
// LINE RECOMBINATION IMPLEMENTATION
// ============================================================================

std::pair<RealVector, RealVector> LineRecombination::crossover(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    std::uniform_real_distribution<double> dist(-extension_factor, 1.0 + extension_factor);
    
    RealVector child1, child2;
    child1.reserve(parent1.size());
    child2.reserve(parent1.size());
    
    for (size_t i = 0; i < parent1.size(); ++i) {
        double alpha1 = dist(rng);
        double alpha2 = dist(rng);
        
        child1.push_back(alpha1 * parent1[i] + (1.0 - alpha1) * parent2[i]);
        child2.push_back(alpha2 * parent1[i] + (1.0 - alpha2) * parent2[i]);
    }
    
    return {child1, child2};
}
