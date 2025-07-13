#include "base_mutation.h"

MutationOperator::MutationOperator(const std::string& name, unsigned seed) 
    : rng(seed), uniform_dist(0.0, 1.0), normal_dist(0.0, 1.0), operator_name(name) {
}

void MutationOperator::validateProbability(double pm, const std::string& methodName) const {
    if (pm < 0.0 || pm > 1.0) {
        throw InvalidParameterException(methodName + " - probability must be in [0,1], got: " + std::to_string(pm));
    }
}

void MutationOperator::validateBounds(const std::vector<double>& lower, 
                                     const std::vector<double>& upper) const {
    if (lower.size() != upper.size()) {
        throw InvalidParameterException("Lower and upper bounds size mismatch: " + 
                         std::to_string(lower.size()) + " vs " + std::to_string(upper.size()));
    }
    
    for (size_t i = 0; i < lower.size(); ++i) {
        if (lower[i] > upper[i]) {
            throw InvalidParameterException("Invalid bounds at index " + std::to_string(i) + 
                             ": lower=" + std::to_string(lower[i]) + 
                             " > upper=" + std::to_string(upper[i]));
        }
    }
}

void MutationOperator::setSeed(unsigned int seed) {
    rng.seed(seed);
}
