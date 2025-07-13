#ifndef ALL_SELECTION_OPERATORS_H
#define ALL_SELECTION_OPERATORS_H

/**
 * @file all_selection_operators.h
 * @brief Unified header for all selection operators
 * 
 * This header provides a convenient way to include all selection operators
 * in a single include statement. It also provides factory functions and
 * utility functions for working with selection operators.
 */

// Base selection operator
#include "base_selection.h"

// Individual selection operators
#include "tournament_selection.h"
#include "roulette_wheel_selection.h"
#include "rank_selection.h"
#include "stochastic_universal_sampling.h"
#include "elitism_selection.h"

#include <memory>
#include <string>
#include <map>
#include <vector>

namespace SelectionOperators {

/**
 * @brief Selection operator types
 */
enum class SelectionType {
    TOURNAMENT,
    ROULETTE_WHEEL,
    RANK,
    STOCHASTIC_UNIVERSAL_SAMPLING,
    ELITISM
};

/**
 * @brief Factory function to create selection operators
 * @param type The type of selection operator to create
 * @param params Optional parameters (e.g., tournament size, selection pressure)
 * @return Unique pointer to the created selection operator
 */
std::unique_ptr<SelectionOperator> createSelectionOperator(SelectionType type, 
                                                          const std::map<std::string, double>& params = {});

/**
 * @brief Get all available selection operator names
 * @return Vector of selection operator names
 */
std::vector<std::string> getAvailableOperators();

/**
 * @brief Demonstrate all selection operators
 * @param population Test population to demonstrate on
 * @param select_count Number of individuals to select in demonstration
 */
void demonstrateAllOperators(const std::vector<Individual>& population, size_t select_count = 5);

/**
 * @brief Get selection operator by name (case-insensitive)
 * @param name Name of the selection operator
 * @param params Optional parameters
 * @return Unique pointer to the created selection operator, or nullptr if not found
 */
std::unique_ptr<SelectionOperator> createByName(const std::string& name,
                                               const std::map<std::string, double>& params = {});

/**
 * @brief Compare performance of different selection operators
 * @param population Test population
 * @param select_count Number of individuals to select
 * @param iterations Number of iterations to run for timing
 */
void compareOperatorPerformance(const std::vector<Individual>& population,
                               size_t select_count = 10,
                               size_t iterations = 1000);

} // namespace SelectionOperators

#endif // ALL_SELECTION_OPERATORS_H
