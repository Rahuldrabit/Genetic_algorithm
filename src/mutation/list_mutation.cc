#include "list_mutation.h"
#include <algorithm>
#include <random>
#include <sstream>

ListMutation::ListMutation(unsigned int seed) : MutationOperator("ListMutation", seed) {
}

void ListMutation::mutate(std::vector<int>& list, 
                         double pm_content, 
                         double pm_size,
                         int minVal, 
                         int maxVal, 
                         size_t minSize, 
                         size_t maxSize) const {
    validateParameters(pm_content, pm_size, minVal, maxVal, minSize, maxSize);
    
    stats.totalMutations++;
    
    try {
        // First mutate content, then size (order matters for statistics)
        mutateContent(list, pm_content, minVal, maxVal);
        mutateSize(list, pm_size, minVal, maxVal, minSize, maxSize);
        
        stats.successfulMutations++;
        
    } catch (const std::exception& e) {
        stats.failedMutations++;
        throw MutationException("List mutation failed: " + std::string(e.what()));
    }
}

void ListMutation::mutateContent(std::vector<int>& list, 
                                double pm_content,
                                int minVal, 
                                int maxVal) const {
    if (list.empty()) {
        return;
    }
    
    std::uniform_int_distribution<int> value_dist(minVal, maxVal);
    
    for (size_t i = 0; i < list.size(); ++i) {
        if (uniform_dist(rng) < pm_content) {
            // Mutate this element
            int newValue;
            do {
                newValue = value_dist(rng);
            } while (newValue == list[i] && minVal < maxVal); // Ensure actual change if possible
            
            list[i] = newValue;
        }
    }
}

void ListMutation::mutateSize(std::vector<int>& list,
                             double pm_size,
                             int minVal,
                             int maxVal,
                             size_t minSize,
                             size_t maxSize) const {
    if (uniform_dist(rng) >= pm_size) {
        return; // No size mutation
    }
    
    size_t currentSize = list.size();
    
    // Determine if we should add or remove elements
    bool canGrow = currentSize < maxSize;
    bool canShrink = currentSize > minSize;
    
    if (!canGrow && !canShrink) {
        return; // Cannot change size
    }
    
    std::uniform_int_distribution<int> value_dist(minVal, maxVal);
    
    if (canGrow && canShrink) {
        // Randomly choose to grow or shrink
        if (uniform_dist(rng) < 0.5) {
            // Add element
            int newValue = value_dist(rng);
            
            // Choose random position to insert
            std::uniform_int_distribution<size_t> pos_dist(0, currentSize);
            size_t position = pos_dist(rng);
            
            list.insert(list.begin() + position, newValue);
        } else {
            // Remove element
            std::uniform_int_distribution<size_t> pos_dist(0, currentSize - 1);
            size_t position = pos_dist(rng);
            
            list.erase(list.begin() + position);
        }
    } else if (canGrow) {
        // Can only grow
        int newValue = value_dist(rng);
        
        // Choose random position to insert
        std::uniform_int_distribution<size_t> pos_dist(0, currentSize);
        size_t position = pos_dist(rng);
        
        list.insert(list.begin() + position, newValue);
    } else {
        // Can only shrink
        std::uniform_int_distribution<size_t> pos_dist(0, currentSize - 1);
        size_t position = pos_dist(rng);
        
        list.erase(list.begin() + position);
    }
}

void ListMutation::validateParameters(double pm_content, 
                                     double pm_size,
                                     int minVal, 
                                     int maxVal,
                                     size_t minSize, 
                                     size_t maxSize) const {
    validateProbability(pm_content, "ListMutation::mutate (pm_content)");
    validateProbability(pm_size, "ListMutation::mutate (pm_size)");
    
    if (minVal > maxVal) {
        std::ostringstream oss;
        oss << "minVal (" << minVal << ") must be <= maxVal (" << maxVal << ")";
        throw InvalidParameterException(oss.str());
    }
    
    if (minSize > maxSize) {
        std::ostringstream oss;
        oss << "minSize (" << minSize << ") must be <= maxSize (" << maxSize << ")";
        throw InvalidParameterException(oss.str());
    }
    
    if (minSize == 0) {
        throw InvalidParameterException("minSize must be at least 1");
    }
}
