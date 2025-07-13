#include "one_point_crossover.h"
#include <stdexcept>

std::pair<BitString, BitString> OnePointCrossover::crossover(const BitString& parent1, const BitString& parent2) {
    try {
        if (parent1.size() != parent2.size()) {
            logError("Parent chromosomes have different sizes");
            return {parent1, parent2};
        }
        
        size_t length = parent1.size();
        if (length <= 1) {
            logOperation("Crossover with length <= 1, returning parents", true);
            return {parent1, parent2};
        }
        
        std::uniform_int_distribution<size_t> dist(1, length - 1);
        size_t crossover_point = dist(rng);
        
        BitString child1 = parent1;
        BitString child2 = parent2;
        
        for (size_t i = crossover_point; i < length; ++i) {
            child1[i] = parent2[i];
            child2[i] = parent1[i];
        }
        
        logOperation("Binary one-point crossover at position " + std::to_string(crossover_point), true);
        return {child1, child2};
        
    } catch (const std::exception& e) {
        logError("Exception during binary crossover: " + std::string(e.what()));
        return {parent1, parent2};
    }
}

std::pair<RealVector, RealVector> OnePointCrossover::crossover(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        logError("Parent chromosomes have different sizes");
        return {parent1, parent2};
    }
    
    size_t length = parent1.size();
    if (length <= 1) {
        logOperation("Crossover with length <= 1, returning parents", true);
        return {parent1, parent2};
    }
    
    std::uniform_int_distribution<size_t> dist(1, length - 1);
    size_t crossover_point = dist(rng);
    
    RealVector child1 = parent1;
    RealVector child2 = parent2;
    
    for (size_t i = crossover_point; i < length; ++i) {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
    
    logOperation("Real-valued one-point crossover at position " + std::to_string(crossover_point), true);
    return {child1, child2};
}

std::pair<IntVector, IntVector> OnePointCrossover::crossover(const IntVector& parent1, const IntVector& parent2) {
    if (parent1.size() != parent2.size()) {
        logError("Parent chromosomes have different sizes");
        return {parent1, parent2};
    }
    
    size_t length = parent1.size();
    if (length <= 1) {
        logOperation("Crossover with length <= 1, returning parents", true);
        return {parent1, parent2};
    }
    
    std::uniform_int_distribution<size_t> dist(1, length - 1);
    size_t crossover_point = dist(rng);
    
    IntVector child1 = parent1;
    IntVector child2 = parent2;
    
    for (size_t i = crossover_point; i < length; ++i) {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
    
    logOperation("Integer one-point crossover at position " + std::to_string(crossover_point), true);
    return {child1, child2};
}
