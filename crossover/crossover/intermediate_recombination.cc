#include "intermediate_recombination.h"
#include <stdexcept>

// ============================================================================
// INTERMEDIATE RECOMBINATION IMPLEMENTATION
// ============================================================================

std::pair<RealVector, RealVector> IntermediateRecombination::crossover(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    RealVector child1, child2;
    child1.reserve(parent1.size());
    child2.reserve(parent1.size());
    
    for (size_t i = 0; i < parent1.size(); ++i) {
        child1.push_back(alpha * parent1[i] + (1.0 - alpha) * parent2[i]);
        child2.push_back(alpha * parent2[i] + (1.0 - alpha) * parent1[i]);
    }
    
    return {child1, child2};
}

RealVector IntermediateRecombination::singleArithmeticRecombination(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    RealVector child = parent1;
    
    // Select random position for arithmetic recombination
    std::uniform_int_distribution<size_t> dist(0, parent1.size() - 1);
    size_t pos = dist(rng);
    
    child[pos] = alpha * parent1[pos] + (1.0 - alpha) * parent2[pos];
    
    return child;
}

RealVector IntermediateRecombination::wholeArithmeticRecombination(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    RealVector child;
    child.reserve(parent1.size());
    
    for (size_t i = 0; i < parent1.size(); ++i) {
        child.push_back(alpha * parent1[i] + (1.0 - alpha) * parent2[i]);
    }
    
    return child;
}
