#pragma once

// Base crossover functionality
#include "ga/crossover/base_crossover.h"

// Vector / Bit-string crossover operators
#include "ga/crossover/one_point_crossover.h"
#include "ga/crossover/two_point_crossover.h"
#include "ga/crossover/multi_point_crossover.h"
#include "ga/crossover/uniform_crossover.h"
#include "ga/crossover/uniform_k_vector_crossover.h"

// Real-valued crossover operators
#include "ga/crossover/blend_crossover.h"
#include "ga/crossover/simulated_binary_crossover.h"
#include "ga/crossover/line_recombination.h"
#include "ga/crossover/intermediate_recombination.h"

// Permutation crossover operators
#include "ga/crossover/cut_and_crossfill_crossover.h"
#include "ga/crossover/partially_mapped_crossover.h"
#include "ga/crossover/edge_crossover.h"
#include "ga/crossover/order_crossover.h"
#include "ga/crossover/cycle_crossover.h"

// Tree and specialized crossover operators
#include "ga/crossover/subtree_crossover.h"
#include "ga/crossover/diploid_recombination.h"
#include "ga/crossover/differential_evolution_crossover.h"
