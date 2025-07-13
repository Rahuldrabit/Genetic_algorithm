#include "self_adaptive_mutation.h"
#include <algorithm>
#include <cmath>
#include <sstream>

SelfAdaptiveMutation::SelfAdaptiveIndividual::SelfAdaptiveIndividual(size_t size, double initial_sigma)
    : genes(size), sigma(initial_sigma) {
    if (initial_sigma <= 0.0) {
        throw BaseMutation::InvalidParameterException("Initial sigma must be positive");
    }
}

SelfAdaptiveMutation::SelfAdaptiveMutation(unsigned int seed) 
    : BaseMutation(seed), normal_dist(0.0, 1.0) {
}

void SelfAdaptiveMutation::mutate(SelfAdaptiveIndividual& individual,
                                  const std::vector<double>& lowerBounds,
                                  const std::vector<double>& upperBounds,
                                  double tau) const {
    if (tau <= 0.0) {
        throw InvalidParameterException("tau must be positive");
    }
    
    validateBounds(lowerBounds, upperBounds, individual.genes.size());
    
    stats.totalMutations++;
    
    try {
        // Step 1: Mutate the strategy parameter (sigma)
        // sigma' = sigma * exp(tau * N(0,1))
        double normalRandom = normal_dist(rng);
        individual.sigma *= std::exp(tau * normalRandom);
        
        // Prevent sigma from becoming too small or too large
        const double MIN_SIGMA = 1e-10;
        const double MAX_SIGMA = 1e10;
        individual.sigma = std::max(MIN_SIGMA, std::min(MAX_SIGMA, individual.sigma));
        
        // Step 2: Mutate the objective variables
        // x'_i = x_i + sigma' * N_i(0,1)
        double totalPerturbation = 0.0;
        for (size_t i = 0; i < individual.genes.size(); ++i) {
            double perturbation = individual.sigma * normal_dist(rng);
            individual.genes[i] += perturbation;
            
            // Apply bounds constraints
            individual.genes[i] = applyBounds(individual.genes[i], 
                                            lowerBounds[i], 
                                            upperBounds[i]);
            
            totalPerturbation += std::abs(perturbation);
        }
        
        // Update statistics
        stats.averagePerturbation = (stats.averagePerturbation * stats.successfulMutations + 
                                   totalPerturbation / individual.genes.size()) / 
                                   (stats.successfulMutations + 1);
        stats.successfulMutations++;
        
    } catch (const std::exception& e) {
        stats.failedMutations++;
        throw MutationException("Self-adaptive mutation failed: " + std::string(e.what()));
    }
}

void SelfAdaptiveMutation::validateBounds(const std::vector<double>& lowerBounds,
                                         const std::vector<double>& upperBounds,
                                         size_t numGenes) const {
    if (lowerBounds.size() != numGenes) {
        std::ostringstream oss;
        oss << "Lower bounds size (" << lowerBounds.size() 
            << ") doesn't match number of genes (" << numGenes << ")";
        throw InvalidParameterException(oss.str());
    }
    
    if (upperBounds.size() != numGenes) {
        std::ostringstream oss;
        oss << "Upper bounds size (" << upperBounds.size() 
            << ") doesn't match number of genes (" << numGenes << ")";
        throw InvalidParameterException(oss.str());
    }
    
    for (size_t i = 0; i < numGenes; ++i) {
        if (lowerBounds[i] >= upperBounds[i]) {
            std::ostringstream oss;
            oss << "Lower bound (" << lowerBounds[i] 
                << ") must be less than upper bound (" << upperBounds[i] 
                << ") for gene " << i;
            throw InvalidParameterException(oss.str());
        }
    }
}

double SelfAdaptiveMutation::applyBounds(double value, double lower, double upper) const {
    return std::max(lower, std::min(upper, value));
}
