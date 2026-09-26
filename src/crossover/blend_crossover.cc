#include "blend_crossover.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

// ============================================================================
// BLEND CROSSOVER (BLX-α) IMPLEMENTATION
// ============================================================================

std::pair<RealVector, RealVector> BlendCrossover::crossover(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    RealVector child1(length), child2(length);
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // For each gene position, perform blend crossover
    for (size_t i = 0; i < length; ++i) {
        double p1_val = parent1[i];
        double p2_val = parent2[i];
        
        // Determine the interval bounds
        double lower = std::min(p1_val, p2_val);
        double upper = std::max(p1_val, p2_val);
        double interval = upper - lower;
        
        // Extend the interval by alpha on both sides
        double extended_lower = lower - alpha * interval;
        double extended_upper = upper + alpha * interval;
        
        // Generate random values within the extended interval
        std::uniform_real_distribution<double> gene_dist(extended_lower, extended_upper);
        child1[i] = gene_dist(rng);
        child2[i] = gene_dist(rng);
    }
    
    return {child1, child2};
}
