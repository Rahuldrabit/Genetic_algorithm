#include "cut_and_crossfill_crossover.h"
#include <unordered_set>
#include <stdexcept>

// ============================================================================
// CUT-AND-CROSSFILL CROSSOVER IMPLEMENTATION
// ============================================================================

std::pair<Permutation, Permutation> CutAndCrossfillCrossover::crossover(const Permutation& parent1, const Permutation& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    if (length <= 1) return {parent1, parent2};
    
    // Select random cut point
    std::uniform_int_distribution<size_t> dist(1, length - 1);
    size_t cut_point = dist(rng);
    
    Permutation child1, child2;
    child1.reserve(length);
    child2.reserve(length);
    
    // Copy first segments
    for (size_t i = 0; i < cut_point; ++i) {
        child1.push_back(parent1[i]);
        child2.push_back(parent2[i]);
    }
    
    // Fill remaining positions from other parent (in order)
    std::unordered_set<int> used1(child1.begin(), child1.end());
    std::unordered_set<int> used2(child2.begin(), child2.end());
    
    for (size_t i = 0; i < length; ++i) {
        if (used1.find(parent2[i]) == used1.end()) {
            child1.push_back(parent2[i]);
            used1.insert(parent2[i]);
        }
        if (used2.find(parent1[i]) == used2.end()) {
            child2.push_back(parent1[i]);
            used2.insert(parent1[i]);
        }
    }
    
    return {child1, child2};
}
