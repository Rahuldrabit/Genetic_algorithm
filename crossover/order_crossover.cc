#include "order_crossover.h"
#include <algorithm>
#include <unordered_set>

std::pair<Permutation, Permutation> OrderCrossover::crossover(const Permutation& parent1, const Permutation& parent2) {
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
    
    Permutation child1(length, -1);
    Permutation child2(length, -1);
    
    // Copy the segment between crossover points
    std::unordered_set<int> used_in_child1, used_in_child2;
    
    for (size_t i = point1; i <= point2; ++i) {
        child1[i] = parent1[i];
        child2[i] = parent2[i];
        used_in_child1.insert(parent1[i]);
        used_in_child2.insert(parent2[i]);
    }
    
    // Fill remaining positions for child1
    size_t child1_pos = (point2 + 1) % length;
    for (size_t i = 0; i < length; ++i) {
        size_t parent2_pos = (point2 + 1 + i) % length;
        int value = parent2[parent2_pos];
        
        if (used_in_child1.find(value) == used_in_child1.end()) {
            while (child1[child1_pos] != -1) {
                child1_pos = (child1_pos + 1) % length;
            }
            child1[child1_pos] = value;
            used_in_child1.insert(value);
        }
    }
    
    // Fill remaining positions for child2
    size_t child2_pos = (point2 + 1) % length;
    for (size_t i = 0; i < length; ++i) {
        size_t parent1_pos = (point2 + 1 + i) % length;
        int value = parent1[parent1_pos];
        
        if (used_in_child2.find(value) == used_in_child2.end()) {
            while (child2[child2_pos] != -1) {
                child2_pos = (child2_pos + 1) % length;
            }
            child2[child2_pos] = value;
            used_in_child2.insert(value);
        }
    }
    
    logOperation("Order crossover (OX) between positions " + 
                std::to_string(point1) + " and " + std::to_string(point2), true);
    return {child1, child2};
}
