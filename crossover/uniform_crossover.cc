#include "uniform_crossover.h"

std::pair<BitString, BitString> UniformCrossover::crossover(const BitString& parent1, const BitString& parent2) {
    if (parent1.size() != parent2.size()) {
        logError("Parent chromosomes have different sizes");
        return {parent1, parent2};
    }
    
    BitString child1 = parent1;
    BitString child2 = parent2;
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < parent1.size(); ++i) {
        if (dist(rng) < probability) {
            child1[i] = parent2[i];
            child2[i] = parent1[i];
        }
    }
    
    logOperation("Binary uniform crossover with probability " + std::to_string(probability), true);
    return {child1, child2};
}

std::pair<RealVector, RealVector> UniformCrossover::crossover(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        logError("Parent chromosomes have different sizes");
        return {parent1, parent2};
    }
    
    RealVector child1 = parent1;
    RealVector child2 = parent2;
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < parent1.size(); ++i) {
        if (dist(rng) < probability) {
            child1[i] = parent2[i];
            child2[i] = parent1[i];
        }
    }
    
    logOperation("Real-valued uniform crossover with probability " + std::to_string(probability), true);
    return {child1, child2};
}

std::pair<IntVector, IntVector> UniformCrossover::crossover(const IntVector& parent1, const IntVector& parent2) {
    if (parent1.size() != parent2.size()) {
        logError("Parent chromosomes have different sizes");
        return {parent1, parent2};
    }
    
    IntVector child1 = parent1;
    IntVector child2 = parent2;
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < parent1.size(); ++i) {
        if (dist(rng) < probability) {
            child1[i] = parent2[i];
            child2[i] = parent1[i];
        }
    }
    
    logOperation("Integer uniform crossover with probability " + std::to_string(probability), true);
    return {child1, child2};
}
