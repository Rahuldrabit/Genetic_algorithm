#include "cycle_crossover.h"
#include <unordered_set>
#include <stdexcept>

// ============================================================================
// CYCLE CROSSOVER IMPLEMENTATION
// ============================================================================

std::vector<std::vector<int>> CycleCrossover::findCycles(const Permutation& parent1, const Permutation& parent2) {
    std::vector<std::vector<int>> cycles;
    std::unordered_set<int> processed;
    
    for (size_t i = 0; i < parent1.size(); ++i) {
        if (processed.find(i) != processed.end()) continue;
        
        std::vector<int> cycle;
        int current = i;
        
        do {
            cycle.push_back(current);
            processed.insert(current);
            
            // Find where parent1[current] appears in parent2
            int target_value = parent1[current];
            for (size_t j = 0; j < parent2.size(); ++j) {
                if (parent2[j] == target_value) {
                    current = j;
                    break;
                }
            }
        } while (current != static_cast<int>(i));
        
        cycles.push_back(cycle);
    }
    
    return cycles;
}

std::pair<Permutation, Permutation> CycleCrossover::crossover(const Permutation& parent1, const Permutation& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    auto cycles = findCycles(parent1, parent2);
    
    Permutation child1 = parent2;
    Permutation child2 = parent1;
    
    // Alternate cycles between parents
    for (size_t i = 0; i < cycles.size(); i += 2) {
        for (int pos : cycles[i]) {
            child1[pos] = parent1[pos];
            child2[pos] = parent2[pos];
        }
    }
    
    return {child1, child2};
}
