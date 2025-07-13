#include "gaussian_mutation.h"
#include <algorithm>

// ============================================================================
// GAUSSIAN MUTATION IMPLEMENTATION
// ============================================================================

void GaussianMutation::mutate(std::vector<double>& chromosome, double pm, double sigma,
                             const std::vector<double>& lowerBounds,
                             const std::vector<double>& upperBounds) const {
    validateProbability(pm, "gaussianMutation");
    validateBounds(lowerBounds, upperBounds);
    
    if (sigma <= 0.0) {
        throw InvalidParameterException("Sigma must be positive, got: " + std::to_string(sigma));
    }
    
    if (chromosome.size() != lowerBounds.size()) {
        throw InvalidParameterException("Chromosome size mismatch with bounds");
    }
    
    stats.totalMutations++;
    
    std::normal_distribution<double> gauss_dist(0.0, sigma);
    
    for (size_t i = 0; i < chromosome.size(); ++i) {
        if (uniform_dist(rng) < pm) {
            double perturbation = gauss_dist(rng);
            chromosome[i] += perturbation;
            
            // Clamp to valid range
            chromosome[i] = std::max(lowerBounds[i], 
                                   std::min(upperBounds[i], chromosome[i]));
            
            stats.successfulMutations++;
        }
    }
}
