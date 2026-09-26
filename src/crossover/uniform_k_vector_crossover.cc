#include "uniform_k_vector_crossover.h"
#include <stdexcept>
#include <algorithm>

// ============================================================================
// UNIFORM K-VECTOR CROSSOVER IMPLEMENTATION
// ============================================================================

std::vector<BitString> UniformKVectorCrossover::crossover(const std::vector<BitString>& parents) {
    if (parents.empty()) {
        throw std::invalid_argument("Parent vector cannot be empty");
    }
    
    operation_count++;
    
    size_t num_parents = parents.size();
    size_t length = parents[0].size();
    
    // Check all parents have same length
    for (const auto& parent : parents) {
        if (parent.size() != length) {
            throw std::invalid_argument("All parents must have the same length");
        }
    }
    
    std::vector<BitString> offspring = parents;
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_int_distribution<size_t> parent_dist(0, num_parents - 1);
    
    for (size_t i = 0; i < length; ++i) {
        if (prob_dist(rng) < swap_probability) {
            // Randomly shuffle the genes at position i among all offspring
            std::vector<bool> genes_at_i;
            for (size_t j = 0; j < num_parents; ++j) {
                genes_at_i.push_back(parents[j][i]);
            }
            
            std::shuffle(genes_at_i.begin(), genes_at_i.end(), rng);
            
            for (size_t j = 0; j < num_parents; ++j) {
                offspring[j][i] = genes_at_i[j];
            }
        }
    }
    
    return offspring;
}

std::vector<RealVector> UniformKVectorCrossover::crossover(const std::vector<RealVector>& parents) {
    if (parents.empty()) {
        throw std::invalid_argument("Parent vector cannot be empty");
    }
    
    operation_count++;
    
    size_t num_parents = parents.size();
    size_t length = parents[0].size();
    
    // Check all parents have same length
    for (const auto& parent : parents) {
        if (parent.size() != length) {
            throw std::invalid_argument("All parents must have the same length");
        }
    }
    
    std::vector<RealVector> offspring = parents;
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    
    for (size_t i = 0; i < length; ++i) {
        if (prob_dist(rng) < swap_probability) {
            // Randomly shuffle the genes at position i among all offspring
            std::vector<double> genes_at_i;
            for (size_t j = 0; j < num_parents; ++j) {
                genes_at_i.push_back(parents[j][i]);
            }
            
            std::shuffle(genes_at_i.begin(), genes_at_i.end(), rng);
            
            for (size_t j = 0; j < num_parents; ++j) {
                offspring[j][i] = genes_at_i[j];
            }
        }
    }
    
    return offspring;
}

std::vector<IntVector> UniformKVectorCrossover::crossover(const std::vector<IntVector>& parents) {
    if (parents.empty()) {
        throw std::invalid_argument("Parent vector cannot be empty");
    }
    
    operation_count++;
    
    size_t num_parents = parents.size();
    size_t length = parents[0].size();
    
    // Check all parents have same length
    for (const auto& parent : parents) {
        if (parent.size() != length) {
            throw std::invalid_argument("All parents must have the same length");
        }
    }
    
    std::vector<IntVector> offspring = parents;
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    
    for (size_t i = 0; i < length; ++i) {
        if (prob_dist(rng) < swap_probability) {
            // Randomly shuffle the genes at position i among all offspring
            std::vector<int> genes_at_i;
            for (size_t j = 0; j < num_parents; ++j) {
                genes_at_i.push_back(parents[j][i]);
            }
            
            std::shuffle(genes_at_i.begin(), genes_at_i.end(), rng);
            
            for (size_t j = 0; j < num_parents; ++j) {
                offspring[j][i] = genes_at_i[j];
            }
        }
    }
    
    return offspring;
}
