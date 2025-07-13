#include "partially_mapped_crossover.h"
#include <algorithm>
#include <unordered_map>

std::pair<Permutation, Permutation> PartiallyMappedCrossover::crossover(const Permutation& parent1, const Permutation& parent2) {
    if (parent1.size() != parent2.size()) {
        logError("Parent chromosomes have different sizes");
        return {parent1, parent2};
    }
    
    size_t length = parent1.size();
    if (length <= 2) {
        logOperation("Crossover with length <= 2, returning parents", true);
        return {parent1, parent2};
    }
    
    // Select two random crossover points
    std::uniform_int_distribution<size_t> dist(0, length - 1);
    size_t point1 = dist(rng);
    size_t point2 = dist(rng);
    
    if (point1 > point2) std::swap(point1, point2);
    
    Permutation child1 = parent1;
    Permutation child2 = parent2;
    
    // Create mapping from the crossing section
    std::unordered_map<int, int> mapping1to2, mapping2to1;
    
    for (size_t i = point1; i <= point2; ++i) {
        // Swap the crossing section
        std::swap(child1[i], child2[i]);
        
        // Build mappings
        mapping1to2[parent1[i]] = parent2[i];
        mapping2to1[parent2[i]] = parent1[i];
    }
    
    // Fix conflicts outside the crossing section for child1
    for (size_t i = 0; i < length; ++i) {
        if (i < point1 || i > point2) {
            int value = child1[i];
            while (mapping1to2.find(value) != mapping1to2.end()) {
                value = mapping1to2[value];
            }
            child1[i] = value;
        }
    }
    
    // Fix conflicts outside the crossing section for child2
    for (size_t i = 0; i < length; ++i) {
        if (i < point1 || i > point2) {
            int value = child2[i];
            while (mapping2to1.find(value) != mapping2to1.end()) {
                value = mapping2to1[value];
            }
            child2[i] = value;
        }
    }
    
    logOperation("Partially mapped crossover (PMX) between positions " + 
                std::to_string(point1) + " and " + std::to_string(point2), true);
    return {child1, child2};
}
