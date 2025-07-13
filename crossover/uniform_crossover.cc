#include "uniform_crossover.h"
#include <stdexcept>

// ============================================================================
// UNIFORM CROSSOVER IMPLEMENTATION
// ============================================================================

std::pair<BitString, BitString> UniformCrossover::crossover(const BitString& parent1, const BitString& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    BitString child1, child2;
    child1.reserve(length);
    child2.reserve(length);
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // For each gene position, randomly choose which parent to take gene from
    for (size_t i = 0; i < length; ++i) {
        if (dist(rng) < probability) {
            // Take from first parent for child1, second parent for child2
            child1.push_back(parent1[i]);
            child2.push_back(parent2[i]);
        } else {
            // Take from second parent for child1, first parent for child2
            child1.push_back(parent2[i]);
            child2.push_back(parent1[i]);
        }
    }
    
    return {child1, child2};
}

std::pair<RealVector, RealVector> UniformCrossover::crossover(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    RealVector child1, child2;
    child1.reserve(length);
    child2.reserve(length);
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // For each gene position, randomly choose which parent to take gene from
    for (size_t i = 0; i < length; ++i) {
        if (dist(rng) < probability) {
            // Take from first parent for child1, second parent for child2
            child1.push_back(parent1[i]);
            child2.push_back(parent2[i]);
        } else {
            // Take from second parent for child1, first parent for child2
            child1.push_back(parent2[i]);
            child2.push_back(parent1[i]);
        }
    }
    
    return {child1, child2};
}

std::pair<IntVector, IntVector> UniformCrossover::crossover(const IntVector& parent1, const IntVector& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    IntVector child1, child2;
    child1.reserve(length);
    child2.reserve(length);
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // For each gene position, randomly choose which parent to take gene from
    for (size_t i = 0; i < length; ++i) {
        if (dist(rng) < probability) {
            // Take from first parent for child1, second parent for child2
            child1.push_back(parent1[i]);
            child2.push_back(parent2[i]);
        } else {
            // Take from second parent for child1, first parent for child2
            child1.push_back(parent2[i]);
            child2.push_back(parent1[i]);
        }
    }
    
    return {child1, child2};
}
