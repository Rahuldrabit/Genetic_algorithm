#include "order_crossover.h"
#include <stdexcept>
#include <algorithm>
#include <unordered_set>

// ============================================================================
// ORDER CROSSOVER (OX) IMPLEMENTATION
// ============================================================================

std::pair<Permutation, Permutation> OrderCrossover::crossover(const Permutation& parent1, const Permutation& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    if (parent1.empty()) {
        return {parent1, parent2};
    }
    
    operation_count++;
    
    // Create both offspring
    Permutation child1 = createOffspring(parent1, parent2);
    Permutation child2 = createOffspring(parent2, parent1);
    
    return {child1, child2};
}

Permutation OrderCrossover::createOffspring(const Permutation& p1, const Permutation& p2) {
    size_t length = p1.size();
    
    if (length <= 2) {
        return p1; // No meaningful crossover for very short permutations
    }
    
    // Select two random cut points
    std::uniform_int_distribution<size_t> dist(0, length - 1);
    size_t cut1 = dist(rng);
    size_t cut2 = dist(rng);
    
    // Ensure cut1 <= cut2
    if (cut1 > cut2) {
        std::swap(cut1, cut2);
    }
    
    // Initialize offspring with -1 (indicating empty positions)
    Permutation offspring(length, -1);
    
    // Copy the segment between cut points from parent1
    std::unordered_set<int> copied_elements;
    for (size_t i = cut1; i <= cut2; ++i) {
        offspring[i] = p1[i];
        copied_elements.insert(p1[i]);
    }
    
    // Fill remaining positions with elements from parent2 in order
    size_t offspring_pos = 0;
    for (size_t i = 0; i < length; ++i) {
        // Skip the copied segment
        if (offspring_pos >= cut1 && offspring_pos <= cut2) {
            offspring_pos = cut2 + 1;
        }
        
        // If we've filled all positions, break
        if (offspring_pos >= length) {
            break;
        }
        
        // If element from p2 is not already copied, add it
        if (copied_elements.find(p2[i]) == copied_elements.end()) {
            offspring[offspring_pos] = p2[i];
            offspring_pos++;
        }
    }
    
    return offspring;
}
