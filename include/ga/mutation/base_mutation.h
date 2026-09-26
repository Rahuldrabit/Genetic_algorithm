#ifndef BASE_MUTATION_H
#define BASE_MUTATION_H

#include <vector>
#include <random>
#include <string>
#include <stdexcept>

// Type definitions
using BitString = std::vector<bool>;
using RealVector = std::vector<double>;
using IntVector = std::vector<int>;
using Permutation = std::vector<int>;

// Base mutation operator class
class MutationOperator {
public:
    // Exception classes for better error handling
    class MutationException : public std::runtime_error {
    public:
        explicit MutationException(const std::string& msg) : std::runtime_error(msg) {}
    };

    class InvalidParameterException : public MutationException {
    public:
        explicit InvalidParameterException(const std::string& msg) 
            : MutationException("Invalid parameter: " + msg) {}
    };

    // Statistics tracking structure
    struct MutationStats {
        size_t totalMutations = 0;
        size_t successfulMutations = 0;
        size_t failedMutations = 0;
        double averagePerturbation = 0.0;
        
        void reset() {
            totalMutations = successfulMutations = failedMutations = 0;
            averagePerturbation = 0.0;
        }
    };

protected:
    mutable std::mt19937 rng;
    mutable std::uniform_real_distribution<double> uniform_dist;
    mutable std::normal_distribution<double> normal_dist;
    mutable MutationStats stats;
    std::string operator_name;
    
    // Helper methods
    void validateProbability(double pm, const std::string& methodName) const;
    void validateBounds(const std::vector<double>& lower, const std::vector<double>& upper) const;
    
public:
    explicit MutationOperator(const std::string& name = "MutationOperator", unsigned seed = std::random_device{}());
    virtual ~MutationOperator() = default;
    
    // Statistics methods
    const MutationStats& getStatistics() const { return stats; }
    void resetStatistics() { stats.reset(); }
    std::string getName() const { return operator_name; }
    void setSeed(unsigned int seed);
};

#endif // BASE_MUTATION_H
