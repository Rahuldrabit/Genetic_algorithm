#ifndef BASE_CROSSOVER_H
#define BASE_CROSSOVER_H

#include <vector>
#include <random>
#include <utility>
#include <functional>
#include <map>
#include <set>
#include <memory>
#include <stdexcept>

// Type definitions
using BitString = std::vector<bool>;
using RealVector = std::vector<double>;
using IntVector = std::vector<int>;
using Permutation = std::vector<int>;

// Tree node structure for GP
struct TreeNode {
    std::string value;
    std::vector<TreeNode*> children;
    
    TreeNode(const std::string& val) : value(val) {}
    ~TreeNode();
    TreeNode* clone() const;
};

// Rule structure for rule-based systems
struct Rule {
    std::vector<int> conditions;
    std::vector<int> actions;
    double fitness;
    
    Rule() : fitness(0.0) {}
};

using RuleSet = std::vector<Rule>;

// Base crossover operator class
class CrossoverOperator {
protected:
    std::mt19937 rng;
    std::string operator_name;
    mutable size_t operation_count;
    mutable size_t error_count;
    
public:
    CrossoverOperator(const std::string& name = "CrossoverOperator", unsigned seed = std::random_device{}()) 
        : rng(seed), operator_name(name), operation_count(0), error_count(0) {}
    virtual ~CrossoverOperator() = default;
    
    // Virtual crossover methods for different data types
    virtual std::pair<RealVector, RealVector> crossover(const RealVector& /* parent1 */, const RealVector& /* parent2 */) {
        throw std::runtime_error("Real vector crossover not implemented for " + operator_name);
    }
    
    virtual std::pair<BitString, BitString> crossover(const BitString& /* parent1 */, const BitString& /* parent2 */) {
        throw std::runtime_error("Bit string crossover not implemented for " + operator_name);
    }
    
    virtual std::pair<IntVector, IntVector> crossover(const IntVector& /* parent1 */, const IntVector& /* parent2 */) {
        throw std::runtime_error("Integer vector crossover not implemented for " + operator_name);
    }
    
    // Statistics methods
    size_t getOperationCount() const { return operation_count; }
    size_t getErrorCount() const { return error_count; }
    double getErrorRate() const { 
        return operation_count > 0 ? static_cast<double>(error_count) / operation_count : 0.0; 
    }
    void resetStatistics() { operation_count = 0; error_count = 0; }
    
    // Getter for operator name
    std::string getName() const { return operator_name; }
};

#endif // BASE_CROSSOVER_H
