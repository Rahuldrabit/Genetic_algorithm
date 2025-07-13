#include "creep_mutation.h"
#include <algorithm>

// ============================================================================
// CREEP MUTATION IMPLEMENTATION
// ============================================================================

void CreepMutation::mutate(std::vector<int>& chromosome, double pm, int stepSize, int minVal, int maxVal) const {
    validateProbability(pm, "creepMutation");
    
    if (minVal > maxVal || stepSize < 0) {
        throw InvalidParameterException("Invalid parameters: minVal=" + std::to_string(minVal) + 
                         ", maxVal=" + std::to_string(maxVal) + 
                         ", stepSize=" + std::to_string(stepSize));
    }
    
    stats.totalMutations++;
    
    std::uniform_int_distribution<int> step_dist(-stepSize, stepSize);
    
    for (size_t i = 0; i < chromosome.size(); ++i) {
        if (uniform_dist(rng) < pm) {
            int newVal = chromosome[i] + step_dist(rng);
            chromosome[i] = std::max(minVal, std::min(maxVal, newVal));
            stats.successfulMutations++;
        }
    }
}
