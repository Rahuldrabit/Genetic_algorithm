#include "random_resetting_mutation.h"

// ============================================================================
// RANDOM RESETTING MUTATION IMPLEMENTATION
// ============================================================================

void RandomResettingMutation::mutate(std::vector<int>& chromosome, double pm, int minVal, int maxVal) const {
    validateProbability(pm, "randomResettingMutation");
    
    if (minVal > maxVal) {
        throw InvalidParameterException("Invalid range: minVal=" + std::to_string(minVal) + 
                         " > maxVal=" + std::to_string(maxVal));
    }
    
    stats.totalMutations++;
    
    std::uniform_int_distribution<int> int_dist(minVal, maxVal);
    
    for (size_t i = 0; i < chromosome.size(); ++i) {
        if (uniform_dist(rng) < pm) {
            chromosome[i] = int_dist(rng);
            stats.successfulMutations++;
        }
    }
}
