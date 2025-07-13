#include "uniform_mutation.h"

// ============================================================================
// UNIFORM MUTATION IMPLEMENTATION
// ============================================================================

void UniformMutation::mutate(std::vector<double>& chromosome, double pm,
                            const std::vector<double>& lowerBounds,
                            const std::vector<double>& upperBounds) const {
    validateProbability(pm, "uniformMutation");
    validateBounds(lowerBounds, upperBounds);
    
    if (chromosome.size() != lowerBounds.size()) {
        throw InvalidParameterException("Chromosome size mismatch with bounds: " + 
                         std::to_string(chromosome.size()) + " vs " + std::to_string(lowerBounds.size()));
    }
    
    stats.totalMutations++;
    
    for (size_t i = 0; i < chromosome.size(); ++i) {
        if (uniform_dist(rng) < pm) {
            std::uniform_real_distribution<double> range_dist(lowerBounds[i], upperBounds[i]);
            chromosome[i] = range_dist(rng);
            stats.successfulMutations++;
        }
    }
}
