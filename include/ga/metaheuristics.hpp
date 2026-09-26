#pragma once

// Core & Common
#include "ga/metaheuristics/common.hpp"
#include "ga/metaheuristics/custom_optimizer.hpp"
#include "ga/metaheuristics/genetic_algorithm_adapter.hpp"

// ACO (All Types)
#include "ga/aco/ant_colony.hpp"
#include "ga/aco/continuous_ant_colony.hpp"

// PSO (All Types)
#include "ga/pso/particle_swarm.hpp"
#include "ga/pso/clpso.hpp"

// GSO (All Types) & GSA
#include "ga/gso/glowworm_swarm.hpp"
#include "ga/gsa/gravitational_search.hpp"

// Shake Perturbation Framework
#include "ga/shake/shake_types.hpp"
#include "ga/shake/shake_operators.hpp"
#include "ga/shake/adaptive_shaker.hpp"

// Fuzzy Subsystem & Controllers
#include "ga/fuzzy/fuzzy_types.hpp"
#include "ga/fuzzy/membership_functions.hpp"
#include "ga/fuzzy/t_norms.hpp"
#include "ga/fuzzy/s_norms.hpp"
#include "ga/fuzzy/complements.hpp"
#include "ga/fuzzy/defuzzification.hpp"
#include "ga/fuzzy/linguistic_variable.hpp"
#include "ga/fuzzy/fuzzy_rule.hpp"
#include "ga/fuzzy/mamdani_system.hpp"
#include "ga/fuzzy/sugeno_system.hpp"
#include "ga/fuzzy/fuzzy_c_means.hpp"
#include "ga/fuzzy/fuzzy_controller.hpp"
#include "ga/fuzzy/fcm_swarm_analyzer.hpp"
#include "ga/fuzzy/fuzzy_algorithm_adapter.hpp"
#include "ga/fuzzy/fuzzy_adaptive_crossover.hpp"

// Hybrid Systems
#include "ga/hybrid/metaheuristic_pipeline.hpp"
#include "ga/hybrid/interleaved_hybrid.hpp"
#include "ga/hybrid/swarm_genetic_hybrid.hpp"

// Large Population / Distributed Systems
#include "ga/algorithms/island_model.hpp"

// Real-Time & Fast Heuristics (<1ms)
#include "ga/metaheuristics/simulated_annealing.hpp"
#include "ga/metaheuristics/hill_climbing.hpp"

// Adaptive & Dynamic Systems
#include "ga/adaptive/adaptive_policy.hpp"
#include "ga/adaptive/dynamic_ga.hpp"

// Constraint-Handling Framework
#include "ga/constraints/constraints.hpp"
#include "ga/constraints/deb_feasibility.hpp"
#include "ga/constraints/adaptive_penalty.hpp"
#include "ga/constraints/repair_operators.hpp"
#include "ga/constraints/constrained_optimizer.hpp"

// High-level API Runners
#include "ga/api/run_optimizers.hpp"
