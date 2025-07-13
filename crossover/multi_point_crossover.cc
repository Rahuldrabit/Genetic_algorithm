#include "multi_point_crossover.h"
#include <stdexcept>
#include <algorithm>
#include <set>

// ============================================================================
// MULTI-POINT CROSSOVER IMPLEMENTATION
// ============================================================================

std::pair<BitString, BitString> MultiPointCrossover::crossover(const BitString& parent1, const BitString& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    if (length <= 1 || num_points >= static_cast<int>(length)) {
        return {parent1, parent2};
    }
    
    // Generate crossover points
    std::vector<size_t> points = generateCrossoverPoints(length);
    
    // Create offspring
    BitString child1 = parent1;
    BitString child2 = parent2;
    
    // Exchange segments alternately
    bool exchange = true;
    for (size_t i = 0; i < points.size(); ++i) {
        size_t start = (i == 0) ? 0 : points[i-1];
        size_t end = points[i];
        
        if (exchange) {
            for (size_t j = start; j < end; ++j) {
                bool temp = child1[j];
                child1[j] = child2[j];
                child2[j] = temp;
            }
        }
        exchange = !exchange;
    }
    
    // Handle the last segment
    if (exchange && !points.empty()) {
        for (size_t j = points.back(); j < length; ++j) {
            bool temp = child1[j];
            child1[j] = child2[j];
            child2[j] = temp;
        }
    }
    
    return {child1, child2};
}

std::pair<RealVector, RealVector> MultiPointCrossover::crossover(const RealVector& parent1, const RealVector& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    if (length <= 1 || num_points >= static_cast<int>(length)) {
        return {parent1, parent2};
    }
    
    // Generate crossover points
    std::vector<size_t> points = generateCrossoverPoints(length);
    
    // Create offspring
    RealVector child1 = parent1;
    RealVector child2 = parent2;
    
    // Exchange segments alternately
    bool exchange = true;
    for (size_t i = 0; i < points.size(); ++i) {
        size_t start = (i == 0) ? 0 : points[i-1];
        size_t end = points[i];
        
        if (exchange) {
            for (size_t j = start; j < end; ++j) {
                std::swap(child1[j], child2[j]);
            }
        }
        exchange = !exchange;
    }
    
    // Handle the last segment
    if (exchange && !points.empty()) {
        for (size_t j = points.back(); j < length; ++j) {
            std::swap(child1[j], child2[j]);
        }
    }
    
    return {child1, child2};
}

std::pair<IntVector, IntVector> MultiPointCrossover::crossover(const IntVector& parent1, const IntVector& parent2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("Parents must have the same length");
    }
    
    operation_count++;
    
    size_t length = parent1.size();
    if (length <= 1 || num_points >= static_cast<int>(length)) {
        return {parent1, parent2};
    }
    
    // Generate crossover points
    std::vector<size_t> points = generateCrossoverPoints(length);
    
    // Create offspring
    IntVector child1 = parent1;
    IntVector child2 = parent2;
    
    // Exchange segments alternately
    bool exchange = true;
    for (size_t i = 0; i < points.size(); ++i) {
        size_t start = (i == 0) ? 0 : points[i-1];
        size_t end = points[i];
        
        if (exchange) {
            for (size_t j = start; j < end; ++j) {
                std::swap(child1[j], child2[j]);
            }
        }
        exchange = !exchange;
    }
    
    // Handle the last segment
    if (exchange && !points.empty()) {
        for (size_t j = points.back(); j < length; ++j) {
            std::swap(child1[j], child2[j]);
        }
    }
    
    return {child1, child2};
}

std::vector<size_t> MultiPointCrossover::generateCrossoverPoints(size_t length) {
    std::set<size_t> point_set;
    std::uniform_int_distribution<size_t> dist(1, length - 1);
    
    // Generate unique crossover points
    while (static_cast<int>(point_set.size()) < num_points && point_set.size() < length - 1) {
        point_set.insert(dist(rng));
    }
    
    // Convert to sorted vector
    std::vector<size_t> points(point_set.begin(), point_set.end());
    return points;
}
