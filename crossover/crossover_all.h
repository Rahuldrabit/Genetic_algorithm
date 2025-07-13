#ifndef CROSSOVER_ALL_H
#define CROSSOVER_ALL_H

// Include the base crossover functionality
#include "crossover_base.h"

// Include all specific crossover operations
#include "one_point_crossover.h"
#include "two_point_crossover.h"
#include "uniform_crossover.h"
#include "intermediate_recombination.h"
#include "blend_crossover.h"
#include "simulated_binary_crossover.h"
#include "order_crossover.h"
#include "partially_mapped_crossover.h"
#include "cycle_crossover.h"

// Factory function to create crossover operators
std::unique_ptr<CrossoverOperator> createCrossoverOperator(const std::string& type, unsigned seed = std::random_device{}());

#endif // CROSSOVER_ALL_H
