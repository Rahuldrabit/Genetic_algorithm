#include "insert_mutation.h"
#include <algorithm>
#include <random>

InsertMutation::InsertMutation(unsigned int seed) : MutationOperator("InsertMutation", seed) {
}

void InsertMutation::mutate(std::vector<int>& permutation, double pm) const {
    validateProbability(pm, "InsertMutation::mutate");
    
    if (permutation.size() < 2) {
        // Cannot perform insert mutation on permutations with less than 2 elements
        return;
    }

    stats.totalMutations++;
    
    try {
        // Apply mutation based on probability
        if (uniform_dist(rng) < pm) {
            performInsert(permutation);
            stats.successfulMutations++;
        }
    } catch (const std::exception& e) {
        stats.failedMutations++;
        throw MutationException("Insert mutation failed: " + std::string(e.what()));
    }
}

void InsertMutation::performInsert(std::vector<int>& permutation) const {
    size_t size = permutation.size();
    
    // Select random element to move
    std::uniform_int_distribution<size_t> index_dist(0, size - 1);
    size_t fromIndex = index_dist(rng);
    
    // Select random destination position
    size_t toIndex = index_dist(rng);
    
    // If same position, no mutation occurs
    if (fromIndex == toIndex) {
        return;
    }
    
    // Store the element to move
    int element = permutation[fromIndex];
    
    // Remove element from original position and insert at new position
    if (fromIndex < toIndex) {
        // Move elements left to fill the gap
        for (size_t i = fromIndex; i < toIndex; ++i) {
            permutation[i] = permutation[i + 1];
        }
        permutation[toIndex] = element;
    } else {
        // Move elements right to make space
        for (size_t i = fromIndex; i > toIndex; --i) {
            permutation[i] = permutation[i - 1];
        }
        permutation[toIndex] = element;
    }
}
