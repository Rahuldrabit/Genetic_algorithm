#include "partially_mapped_crossover.h"
#include <map>
#include <stdexcept>

// ============================================================================
// PARTIALLY MAPPED CROSSOVER (PMX) IMPLEMENTATION
// ============================================================================

std::pair<Permutation, Permutation> PartiallyMappedCrossover::crossover(const Permutation& parent1, const Permutation& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    if (length <= 2) return {parent1, parent2};
    
    // Select two random crossover points
    std::uniform_int_distribution<size_t> dist(0, length - 1);
    size_t point1 = dist(rng);
    size_t point2 = dist(rng);
    
    if (point1 > point2) std::swap(point1, point2);
    
    Permutation child1(length, -1);
    Permutation child2(length, -1);
    
    // Copy mapping sections and build mappings
    std::map<int, int> mapping1, mapping2;
    for (size_t i = point1; i <= point2; ++i) {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
        mapping1[parent2[i]] = parent1[i];
        mapping2[parent1[i]] = parent2[i];
    }
    
    // Fill remaining positions using mappings
    for (size_t i = 0; i < length; ++i) {
        if (i < point1 || i > point2) {
            // For child1
            int val1 = parent1[i];
            while (mapping1.find(val1) != mapping1.end()) {
                val1 = mapping1[val1];
            }
            child1[i] = val1;
            
            // For child2
            int val2 = parent2[i];
            while (mapping2.find(val2) != mapping2.end()) {
                val2 = mapping2[val2];
            }
            child2[i] = val2;
        }
    }
    
    return {child1, child2};
}
