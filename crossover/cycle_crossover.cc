#include "cycle_crossover.h"
#include <algorithm>
#include <unordered_set>

std::vector<std::vector<int>> CycleCrossover::findCycles(const Permutation& parent1, const Permutation& parent2) {
    std::vector<std::vector<int>> cycles;
    std::unordered_set<int> visited;
    
    for (size_t i = 0; i < parent1.size(); ++i) {
        if (visited.find(i) != visited.end()) continue;
        
        std::vector<int> cycle;
        size_t current = i;
        
        do {
            cycle.push_back(current);
            visited.insert(current);
            
            // Find where parent1[current] appears in parent2
            auto it = std::find(parent2.begin(), parent2.end(), parent1[current]);
            current = std::distance(parent2.begin(), it);
            
        } while (visited.find(current) == visited.end());
        
        if (!cycle.empty()) {
            cycles.push_back(cycle);
        }
    }
    
    return cycles;
}

std::pair<Permutation, Permutation> CycleCrossover::crossover(const Permutation& parent1, const Permutation& parent2) {
    if (parent1.size() != parent2.size()) {
        logError("Parent chromosomes have different sizes");
        return {parent1, parent2};
    }
    
    Permutation child1 = parent1;
    Permutation child2 = parent2;
    
    std::vector<std::vector<int>> cycles = findCycles(parent1, parent2);
    
    // Alternate cycles between parents
    for (size_t cycle_idx = 0; cycle_idx < cycles.size(); ++cycle_idx) {
        if (cycle_idx % 2 == 1) { // Odd cycles: swap values
            for (int pos : cycles[cycle_idx]) {
                std::swap(child1[pos], child2[pos]);
            }
        }
        // Even cycles: keep as is
    }
    
    logOperation("Cycle crossover (CX) with " + std::to_string(cycles.size()) + " cycles found", true);
    return {child1, child2};
}
