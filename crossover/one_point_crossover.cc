#include "one_point_crossover.h"
#include <stdexcept>

// ============================================================================
// ONE-POINT CROSSOVER IMPLEMENTATION
// ============================================================================

std::pair<BitString, BitString> OnePointCrossover::crossover(const BitString& parent1, const BitString& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    if (length <= 1) {
        return {parent1, parent2};
    }
    
    // Select random crossover point (1 to length-1)
    std::uniform_int_distribution<size_t> dist(1, length - 1);
    size_t crossover_point = dist(rng);
    
    // Create offspring
    BitString child1 = parent1;
    BitString child2 = parent2;
    
    // Exchange genetic material after crossover point
    for (size_t i = crossover_point; i < length; ++i) {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
    
    return {child1, child2};
}

std::pair<RealVector, RealVector> OnePointCrossover::crossover(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    if (length <= 1) {
        return {parent1, parent2};
    }
    
    // Select random crossover point (1 to length-1)
    std::uniform_int_distribution<size_t> dist(1, length - 1);
    size_t crossover_point = dist(rng);
    
    // Create offspring
    RealVector child1 = parent1;
    RealVector child2 = parent2;
    
    // Exchange genetic material after crossover point
    for (size_t i = crossover_point; i < length; ++i) {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
    
    return {child1, child2};
}

std::pair<IntVector, IntVector> OnePointCrossover::crossover(const IntVector& parent1, const IntVector& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    if (length <= 1) {
        return {parent1, parent2};
    }
    
    // Select random crossover point (1 to length-1)
    std::uniform_int_distribution<size_t> dist(1, length - 1);
    size_t crossover_point = dist(rng);
    
    // Create offspring
    IntVector child1 = parent1;
    IntVector child2 = parent2;
    
    // Exchange genetic material after crossover point
    for (size_t i = crossover_point; i < length; ++i) {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
    
    return {child1, child2};
}
