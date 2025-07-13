#ifndef ALL_MUTATION_OPERATORS_H
#define ALL_MUTATION_OPERATORS_H

/**
 * @file all_mutation_operators.h
 * @brief Main header file that includes all mutation operators
 * 
 * This header provides convenient access to all mutation operators
 * implemented in the genetic algorithm library. Each operator is
 * implemented in its own separate file for better modularity.
 */

// Base mutation functionality
#include "base_mutation.h"

// Binary representation mutation operators
#include "bit_flip_mutation.h"

// Integer representation mutation operators
#include "random_resetting_mutation.h"
#include "creep_mutation.h"

// Real-valued representation mutation operators
#include "uniform_mutation.h"
#include "gaussian_mutation.h"

// Permutation representation mutation operators
#include "swap_mutation.h"
#include "inversion_mutation.h"
#include "insert_mutation.h"
#include "scramble_mutation.h"

// Self-adaptive mutation operators
#include "self_adaptive_mutation.h"

// Variable-length list mutation operators
#include "list_mutation.h"

#include <memory>
#include <string>
#include <vector>
#include <algorithm>

/**
 * @namespace MutationOperators
 * @brief Factory functions and utilities for mutation operators
 */
namespace MutationOperators {
    /**
     * @brief Factory function to create mutation operators by name
     * @param name The name of the mutation operator
     * @param seed Random seed for reproducibility
     * @return Unique pointer to the created mutation operator
     */
    std::unique_ptr<MutationOperator> create(const std::string& name, unsigned seed = std::random_device{}());
    
    /**
     * @brief Get available mutation operators for a specific encoding type
     * @param encoding_type The type of encoding (binary, real, integer, permutation)
     * @return Vector of available mutation operator names
     */
    std::vector<std::string> getAvailableOperators(const std::string& encoding_type = "all");
    
    /**
     * @brief Check if a mutation operator name is valid
     * @param name The name to check
     * @return True if the operator exists
     */
    bool isValidOperator(const std::string& name);
}

#endif // ALL_MUTATION_OPERATORS_H
