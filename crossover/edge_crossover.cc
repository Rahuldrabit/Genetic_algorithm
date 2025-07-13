#include "edge_crossover.h"
#include <unordered_set>
#include <stdexcept>
#include <climits>

// ============================================================================
// EDGE CROSSOVER IMPLEMENTATION
// ============================================================================

std::map<int, std::set<int>> EdgeCrossover::buildEdgeTable(const Permutation& parent1, const Permutation& parent2) {
    std::map<int, std::set<int>> edge_table;
    
    // Add edges from parent1
    for (size_t i = 0; i < parent1.size(); ++i) {
        int current = parent1[i];
        int prev = parent1[(i - 1 + parent1.size()) % parent1.size()];
        int next = parent1[(i + 1) % parent1.size()];
        
        edge_table[current].insert(prev);
        edge_table[current].insert(next);
    }
    
    // Add edges from parent2
    for (size_t i = 0; i < parent2.size(); ++i) {
        int current = parent2[i];
        int prev = parent2[(i - 1 + parent2.size()) % parent2.size()];
        int next = parent2[(i + 1) % parent2.size()];
        
        edge_table[current].insert(prev);
        edge_table[current].insert(next);
    }
    
    return edge_table;
}

Permutation EdgeCrossover::performCrossover(const Permutation& parent1, const Permutation& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    auto edge_table = buildEdgeTable(parent1, parent2);
    Permutation child;
    child.reserve(parent1.size());
    
    // Start with random city from parent1
    std::uniform_int_distribution<size_t> dist(0, parent1.size() - 1);
    int current = parent1[dist(rng)];
    child.push_back(current);
    
    std::unordered_set<int> visited;
    visited.insert(current);
    
    while (child.size() < parent1.size()) {
        // Remove current city from all edge lists
        for (auto& [city, edges] : edge_table) {
            edges.erase(current);
        }
        
        // Find next city with minimum edge count
        int next_city = -1;
        int min_edges = INT_MAX;
        
        for (int neighbor : edge_table[current]) {
            if (visited.find(neighbor) == visited.end()) {
                int edge_count = edge_table[neighbor].size();
                if (edge_count < min_edges) {
                    min_edges = edge_count;
                    next_city = neighbor;
                }
            }
        }
        
        // If no connected city found, pick random unvisited
        if (next_city == -1) {
            for (const auto& [city, edges] : edge_table) {
                if (visited.find(city) == visited.end()) {
                    next_city = city;
                    break;
                }
            }
        }
        
        current = next_city;
        child.push_back(current);
        visited.insert(current);
    }
    
    return child;
}
