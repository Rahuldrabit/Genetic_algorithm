#ifndef ALL_CROSSOVER_OPERATORS_H
#define ALL_CROSSOVER_OPERATORS_H

/**
 * @file all_crossover_operators.h
 * @brief Main header file that includes all crossover operators
 * 
 * This header provides convenient access to all crossover operators
 * implemented in the genetic algorithm library. Each operator is
 * implemented in its own separate file for better modularity.
 */

// Base crossover functionality
#include "base_crossover.h"

// Bit-string/Vector crossover operators
#include "one_point_crossover.h"
#include "two_point_crossover.h"
#include "multi_point_crossover.h"
#include "uniform_crossover.h"
#include "uniform_k_vector_crossover.h"

// Real-valued crossover operators
#include "blend_crossover.h"
#include "simulated_binary_crossover.h"
#include "line_recombination.h"
#include "intermediate_recombination.h"

// Permutation crossover operators
#include "cut_and_crossfill_crossover.h"
#include "partially_mapped_crossover.h"
#include "edge_crossover.h"
#include "order_crossover.h"
#include "cycle_crossover.h"

// Tree-based crossover operators
#include "subtree_crossover.h"

// Specialized crossover operators
#include "diploid_recombination.h"
#include "differential_evolution_crossover.h"

#include <memory>
#include <string>
#include <vector>

/**
 * @namespace crossover
 * @brief Utility functions for creating crossover operators
 */
namespace crossover {
    
    /**
     * @brief Factory function to create crossover operators by name
     * @param name Name of the crossover operator
     * @return Unique pointer to the crossover operator
     */
    std::unique_ptr<CrossoverOperator> CreateCrossoverOperator(const std::string& name);
    
    /**
     * @brief Get list of available crossover operator names
     * @return Vector of operator names
     */
    std::vector<std::string> GetAvailableCrossoverTypes();
    
    /**
     * @brief Check if an operator name is valid
     * @param name Operator name to check
     * @return True if valid, false otherwise
     */
    bool IsValidOperator(const std::string& name);
}

#endif // ALL_CROSSOVER_OPERATORS_H
