#include "two_point_crossover.h"
#include <algorithm>

std::pair<BitString, BitString> TwoPointCrossover::crossover(const BitString& parent1, const BitString& parent2) {
    if (parent1.size() != parent2.size()) {
        logError("Parent chromosomes have different sizes");
        return {parent1, parent2};
    }
    
    size_t length = parent1.size();
    if (length <= 2) {
        logOperation("Crossover with length <= 2, returning parents", true);
        return {parent1, parent2};
    }
    
    std::uniform_int_distribution<size_t> dist(0, length - 1);
    size_t point1 = dist(rng);
    size_t point2 = dist(rng);
    
    if (point1 > point2) std::swap(point1, point2);
    if (point1 == point2) {
        point2 = (point2 + 1) % length;
        if (point1 > point2) std::swap(point1, point2);
    }
    
    BitString child1 = parent1;
    BitString child2 = parent2;
    
    for (size_t i = point1; i <= point2; ++i) {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
    
    logOperation("Binary two-point crossover between positions " + 
                std::to_string(point1) + " and " + std::to_string(point2), true);
    return {child1, child2};
}

std::pair<RealVector, RealVector> TwoPointCrossover::crossover(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        logError("Parent chromosomes have different sizes");
        return {parent1, parent2};
    }
    
    size_t length = parent1.size();
    if (length <= 2) {
        logOperation("Crossover with length <= 2, returning parents", true);
        return {parent1, parent2};
    }
    
    std::uniform_int_distribution<size_t> dist(0, length - 1);
    size_t point1 = dist(rng);
    size_t point2 = dist(rng);
    
    if (point1 > point2) std::swap(point1, point2);
    if (point1 == point2) {
        point2 = (point2 + 1) % length;
        if (point1 > point2) std::swap(point1, point2);
    }
    
    RealVector child1 = parent1;
    RealVector child2 = parent2;
    
    for (size_t i = point1; i <= point2; ++i) {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
    
    logOperation("Real-valued two-point crossover between positions " + 
                std::to_string(point1) + " and " + std::to_string(point2), true);
    return {child1, child2};
}

std::pair<IntVector, IntVector> TwoPointCrossover::crossover(const IntVector& parent1, const IntVector& parent2) {
    if (parent1.size() != parent2.size()) {
        logError("Parent chromosomes have different sizes");
        return {parent1, parent2};
    }
    
    size_t length = parent1.size();
    if (length <= 2) {
        logOperation("Crossover with length <= 2, returning parents", true);
        return {parent1, parent2};
    }
    
    std::uniform_int_distribution<size_t> dist(0, length - 1);
    size_t point1 = dist(rng);
    size_t point2 = dist(rng);
    
    if (point1 > point2) std::swap(point1, point2);
    if (point1 == point2) {
        point2 = (point2 + 1) % length;
        if (point1 > point2) std::swap(point1, point2);
    }
    
    IntVector child1 = parent1;
    IntVector child2 = parent2;
    
    for (size_t i = point1; i <= point2; ++i) {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
    
    logOperation("Integer two-point crossover between positions " + 
                std::to_string(point1) + " and " + std::to_string(point2), true);
    return {child1, child2};
}
