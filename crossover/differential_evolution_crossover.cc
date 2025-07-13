#include "differential_evolution_crossover.h"
#include <stdexcept>

// ============================================================================
// DIFFERENTIAL EVOLUTION CROSSOVER IMPLEMENTATION
// ============================================================================

RealVector DifferentialEvolutionCrossover::performCrossover(const RealVector& target, const RealVector& mutant) {
    if (target.size() != mutant.size()) {
        throw std::invalid_argument("Target and mutant must have the same length");
    }
    
    operation_count++;
    
    RealVector trial = target;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<size_t> pos_dist(0, target.size() - 1);
    
    // Ensure at least one parameter comes from the mutant
    size_t forced_index = pos_dist(rng);
    
    for (size_t i = 0; i < target.size(); ++i) {
        if (dist(rng) < crossover_rate || i == forced_index) {
            trial[i] = mutant[i];
        }
    }
    
    return trial;
}
