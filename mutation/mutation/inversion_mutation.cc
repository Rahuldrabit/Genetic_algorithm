#include "inversion_mutation.h"
#include <algorithm>

// ============================================================================
// INVERSION MUTATION IMPLEMENTATION
// ============================================================================

void InversionMutation::mutate(std::vector<int>& permutation, double pm) const {
    validateProbability(pm, "inversionMutation");
    
    stats.totalMutations++;
    
    if (uniform_dist(rng) < pm && permutation.size() > 1) {
        std::uniform_int_distribution<size_t> pos_dist(0, permutation.size() - 1);
        size_t start = pos_dist(rng);
        size_t end = pos_dist(rng);
        
        if (start > end) {
            std::swap(start, end);
        }
        
        // Reverse the subsequence
        std::reverse(permutation.begin() + start, 
                    permutation.begin() + end + 1);
        
        stats.successfulMutations++;
    }
}
