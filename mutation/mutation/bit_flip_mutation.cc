#include "bit_flip_mutation.h"

// ============================================================================
// BIT FLIP MUTATION IMPLEMENTATION
// ============================================================================

void BitFlipMutation::mutate(std::vector<bool>& chromosome, double pm) const {
    validateProbability(pm, "bitFlipMutation");
    
    stats.totalMutations++;
    
    for (size_t i = 0; i < chromosome.size(); ++i) {
        if (uniform_dist(rng) < pm) {
            chromosome[i] = !chromosome[i];
            stats.successfulMutations++;
        }
    }
}

void BitFlipMutation::mutate(std::string& binaryString, double pm) const {
    validateProbability(pm, "bitFlipMutation(string)");
    
    stats.totalMutations++;
    
    for (size_t i = 0; i < binaryString.size(); ++i) {
        if (uniform_dist(rng) < pm) {
            binaryString[i] = (binaryString[i] == '0') ? '1' : '0';
            stats.successfulMutations++;
        }
    }
}
