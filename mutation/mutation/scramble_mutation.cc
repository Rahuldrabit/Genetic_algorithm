#include "scramble_mutation.h"
#include <algorithm>
#include <random>

ScrambleMutation::ScrambleMutation(unsigned int seed) : BaseMutation(seed) {
}

void ScrambleMutation::mutate(std::vector<int>& permutation, double pm) const {
    validateProbability(pm, "ScrambleMutation::mutate");
    
    if (permutation.size() < 2) {
        // Cannot perform scramble mutation on permutations with less than 2 elements
        return;
    }

    stats.totalMutations++;
    
    try {
        // Apply mutation based on probability
        if (uniform_dist(rng) < pm) {
            performScramble(permutation);
            stats.successfulMutations++;
        }
    } catch (const std::exception& e) {
        stats.failedMutations++;
        throw MutationException("Scramble mutation failed: " + std::string(e.what()));
    }
}

void ScrambleMutation::performScramble(std::vector<int>& permutation) const {
    size_t size = permutation.size();
    
    // Select two random points to define the scramble region
    std::uniform_int_distribution<size_t> index_dist(0, size - 1);
    size_t point1 = index_dist(rng);
    size_t point2 = index_dist(rng);
    
    // Ensure point1 <= point2
    if (point1 > point2) {
        std::swap(point1, point2);
    }
    
    // If points are the same or adjacent, extend the region
    if (point2 - point1 < 1) {
        if (point2 < size - 1) {
            point2++;
        } else if (point1 > 0) {
            point1--;
        } else {
            // Single element, no scrambling possible
            return;
        }
    }
    
    // Scramble the elements between point1 and point2 (inclusive)
    std::shuffle(permutation.begin() + point1, 
                 permutation.begin() + point2 + 1, 
                 rng);
}
