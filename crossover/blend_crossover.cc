#include "blend_crossover.h"
#include <algorithm>

std::pair<RealVector, RealVector> BlendCrossover::crossover(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        logError("Parent chromosomes have different sizes");
        return {parent1, parent2};
    }
    
    RealVector child1, child2;
    child1.reserve(parent1.size());
    child2.reserve(parent1.size());
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < parent1.size(); ++i) {
        double min_val = std::min(parent1[i], parent2[i]);
        double max_val = std::max(parent1[i], parent2[i]);
        double range = max_val - min_val;
        
        double lower_bound = min_val - alpha * range;
        double upper_bound = max_val + alpha * range;
        
        std::uniform_real_distribution<double> blend_dist(lower_bound, upper_bound);
        
        child1.push_back(blend_dist(rng));
        child2.push_back(blend_dist(rng));
    }
    
    logOperation("Blend crossover (BLX-α) with alpha " + std::to_string(alpha), true);
    return {child1, child2};
}
