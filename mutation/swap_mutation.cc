#include "swap_mutation.h"
#include <algorithm>

// ============================================================================
// SWAP MUTATION IMPLEMENTATION
// ============================================================================

void SwapMutation::mutate(std::vector<int>& permutation, double pm) const {
    validateProbability(pm, "swapMutation");
    
    if (permutation.empty()) {
        return;
    }
    
    stats.totalMutations++;
    
    if (uniform_dist(rng) < pm && permutation.size() > 1) {
        std::uniform_int_distribution<size_t> pos_dist(0, permutation.size() - 1);
        size_t pos1 = pos_dist(rng);
        size_t pos2 = pos_dist(rng);
        
        // Ensure different positions
        size_t attempts = 0;
        while (pos1 == pos2 && attempts < 100) {
            pos2 = pos_dist(rng);
            attempts++;
        }
        
        if (pos1 != pos2) {
            std::swap(permutation[pos1], permutation[pos2]);
            stats.successfulMutations++;
        }
    }
}
