#ifndef CROSSOVER_BASE_H
#define CROSSOVER_BASE_H

#include <vector>
#include <random>
#include <utility>
#include <functional>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <memory>

// ============================================================================
// LOGGING SYSTEM
// ============================================================================

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    CRITICAL = 4
};

class CrossoverLogger {
private:
    static std::unique_ptr<CrossoverLogger> instance;
    std::ofstream log_file;
    LogLevel min_log_level;
    bool console_output;
    std::string log_filename;
    size_t max_file_size;
    size_t current_file_size;
    
    CrossoverLogger();
    std::string getCurrentTimestamp();
    std::string logLevelToString(LogLevel level);
    void rotateLogFile();
    
public:
    static CrossoverLogger& getInstance();
    
    void initialize(const std::string& filename = "crossover.log", 
                   LogLevel level = LogLevel::INFO, 
                   bool console = true,
                   size_t max_size = 10 * 1024 * 1024); // 10MB default
    
    void log(LogLevel level, const std::string& message, 
             const std::string& function = "", int line = 0);
    
    void logError(const std::string& message, const std::string& function = "", int line = 0);
    void logWarning(const std::string& message, const std::string& function = "", int line = 0);
    void logInfo(const std::string& message, const std::string& function = "", int line = 0);
    void logDebug(const std::string& message, const std::string& function = "", int line = 0);
    
    void setLogLevel(LogLevel level) { min_log_level = level; }
    void enableConsoleOutput(bool enable) { console_output = enable; }
    
    ~CrossoverLogger();
};

// Logging macros for convenience
#define LOG_ERROR(msg) CrossoverLogger::getInstance().logError(msg, __FUNCTION__, __LINE__)
#define LOG_WARNING(msg) CrossoverLogger::getInstance().logWarning(msg, __FUNCTION__, __LINE__)
#define LOG_INFO(msg) CrossoverLogger::getInstance().logInfo(msg, __FUNCTION__, __LINE__)
#define LOG_DEBUG(msg) CrossoverLogger::getInstance().logDebug(msg, __FUNCTION__, __LINE__)

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

// Neural network weight structure
struct NeuralNetwork {
    std::vector<std::vector<double>> weights;  // weights[layer][neuron]
    std::vector<std::vector<bool>> connections; // connectivity matrix
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
    
    void logOperation(const std::string& operation, bool success = true) const;
    void logError(const std::string& error_msg) const;
    
public:
    CrossoverOperator(const std::string& name = "CrossoverOperator", unsigned seed = std::random_device{}()) 
        : rng(seed), operator_name(name), operation_count(0), error_count(0) {}
    virtual ~CrossoverOperator() = default;
    
    // Statistics methods
    size_t getOperationCount() const { return operation_count; }
    size_t getErrorCount() const { return error_count; }
    double getErrorRate() const { 
        return operation_count > 0 ? static_cast<double>(error_count) / operation_count : 0.0; 
    }
    void resetStatistics() { operation_count = 0; error_count = 0; }
};

#endif // CROSSOVER_BASE_H
