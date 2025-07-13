#include "two_point_crossover.h"
#include <stdexcept>
#include <algorithm>

// ============================================================================
// TWO-POINT CROSSOVER IMPLEMENTATION
// ============================================================================

std::pair<BitString, BitString> TwoPointCrossover::crossover(const BitString& parent1, const BitString& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    if (length <= 2) {
        return {parent1, parent2};
    }
    
    // Select two random crossover points
    std::uniform_int_distribution<size_t> dist(1, length - 1);
    size_t point1 = dist(rng);
    size_t point2 = dist(rng);
    
    // Ensure point1 < point2
    if (point1 > point2) {
        std::swap(point1, point2);
    }
    
    // If points are the same, make it one-point crossover
    if (point1 == point2) {
        point2 = std::min(point1 + 1, length - 1);
    }
    
    // Create offspring
    BitString child1 = parent1;
    BitString child2 = parent2;
    
    // Exchange genetic material between crossover points
    for (size_t i = point1; i < point2; ++i) {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
    
    return {child1, child2};
}

std::pair<RealVector, RealVector> TwoPointCrossover::crossover(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    if (length <= 2) {
        return {parent1, parent2};
    }
    
    // Select two random crossover points
    std::uniform_int_distribution<size_t> dist(1, length - 1);
    size_t point1 = dist(rng);
    size_t point2 = dist(rng);
    
    // Ensure point1 < point2
    if (point1 > point2) {
        std::swap(point1, point2);
    }
    
    // If points are the same, make it one-point crossover
    if (point1 == point2) {
        point2 = std::min(point1 + 1, length - 1);
    }
    
    // Create offspring
    RealVector child1 = parent1;
    RealVector child2 = parent2;
    
    // Exchange genetic material between crossover points
    for (size_t i = point1; i < point2; ++i) {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
    
    return {child1, child2};
}

std::pair<IntVector, IntVector> TwoPointCrossover::crossover(const IntVector& parent1, const IntVector& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    if (length <= 2) {
        return {parent1, parent2};
    }
    
    // Select two random crossover points
    std::uniform_int_distribution<size_t> dist(1, length - 1);
    size_t point1 = dist(rng);
    size_t point2 = dist(rng);
    
    // Ensure point1 < point2
    if (point1 > point2) {
        std::swap(point1, point2);
    }
    
    // If points are the same, make it one-point crossover
    if (point1 == point2) {
        point2 = std::min(point1 + 1, length - 1);
    }
    
    // Create offspring
    IntVector child1 = parent1;
    IntVector child2 = parent2;
    
    // Exchange genetic material between crossover points
    for (size_t i = point1; i < point2; ++i) {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
    
    return {child1, child2};
}
