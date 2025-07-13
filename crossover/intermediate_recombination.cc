#include "intermediate_recombination.h"

std::pair<RealVector, RealVector> IntermediateRecombination::crossover(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        logError("Parent chromosomes have different sizes");
        return {parent1, parent2};
    }
    
    RealVector child1, child2;
    child1.reserve(parent1.size());
    child2.reserve(parent1.size());
    
    std::uniform_real_distribution<double> dist(-alpha, 1.0 + alpha);
    
    for (size_t i = 0; i < parent1.size(); ++i) {
        double lambda = dist(rng);
        child1.push_back(lambda * parent1[i] + (1.0 - lambda) * parent2[i]);
        
        // Generate different lambda for second child
        lambda = dist(rng);
        child2.push_back(lambda * parent1[i] + (1.0 - lambda) * parent2[i]);
    }
    
    logOperation("Intermediate recombination with alpha " + std::to_string(alpha), true);
    return {child1, child2};
}

RealVector IntermediateRecombination::singleArithmeticRecombination(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        logError("Parent chromosomes have different sizes");
        return parent1;
    }
    
    RealVector child = parent1;
    
    // Select random position for arithmetic recombination
    std::uniform_int_distribution<size_t> pos_dist(0, parent1.size() - 1);
    size_t pos = pos_dist(rng);
    
    child[pos] = alpha * parent1[pos] + (1.0 - alpha) * parent2[pos];
    
    logOperation("Single arithmetic recombination at position " + std::to_string(pos), true);
    return child;
}

RealVector IntermediateRecombination::wholeArithmeticRecombination(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        logError("Parent chromosomes have different sizes");
        return parent1;
    }
    
    RealVector child;
    child.reserve(parent1.size());
    
    for (size_t i = 0; i < parent1.size(); ++i) {
        child.push_back(alpha * parent1[i] + (1.0 - alpha) * parent2[i]);
    }
    
    logOperation("Whole arithmetic recombination with alpha " + std::to_string(alpha), true);
    return child;
}
