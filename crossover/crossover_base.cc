#include "crossover_base.h"
#include <climits>
#include <cstdio>  // for std::rename

// ============================================================================
// LOGGING SYSTEM IMPLEMENTATION
// ============================================================================

std::unique_ptr<CrossoverLogger> CrossoverLogger::instance = nullptr;

CrossoverLogger::CrossoverLogger() 
    : min_log_level(LogLevel::INFO), console_output(true), max_file_size(10 * 1024 * 1024), current_file_size(0) {
}

CrossoverLogger& CrossoverLogger::getInstance() {
    if (!instance) {
        instance = std::unique_ptr<CrossoverLogger>(new CrossoverLogger());
    }
    return *instance;
}

void CrossoverLogger::initialize(const std::string& filename, LogLevel level, bool console, size_t max_size) {
    log_filename = filename;
    min_log_level = level;
    console_output = console;
    max_file_size = max_size;
    
    if (log_file.is_open()) {
        log_file.close();
    }
    
    log_file.open(log_filename, std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Warning: Could not open log file: " << log_filename << std::endl;
    } else {
        current_file_size = log_file.tellp();
    }
}

std::string CrossoverLogger::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

std::string CrossoverLogger::logLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        default: return "UNKNOWN";
    }
}

void CrossoverLogger::rotateLogFile() {
    if (log_file.is_open()) {
        log_file.close();
    }
    
    // Rename current log file
    std::string backup_name = log_filename + ".bak";
    std::rename(log_filename.c_str(), backup_name.c_str());
    
    // Open new log file
    log_file.open(log_filename, std::ios::app);
    current_file_size = 0;
    
    if (log_file.is_open()) {
        log_file << "[" << getCurrentTimestamp() << "] [INFO] Log file rotated" << std::endl;
    }
}

void CrossoverLogger::log(LogLevel level, const std::string& message, const std::string& function, int line) {
    if (level < min_log_level) return;
    
    std::stringstream log_entry;
    log_entry << "[" << getCurrentTimestamp() << "] "
              << "[" << logLevelToString(level) << "] ";
    
    if (!function.empty()) {
        log_entry << "[" << function;
        if (line > 0) {
            log_entry << ":" << line;
        }
        log_entry << "] ";
    }
    
    log_entry << message;
    
    std::string full_message = log_entry.str();
    
    // Console output
    if (console_output) {
        if (level >= LogLevel::ERROR) {
            std::cerr << full_message << std::endl;
        } else {
            std::cout << full_message << std::endl;
        }
    }
    
    // File output
    if (log_file.is_open()) {
        log_file << full_message << std::endl;
        log_file.flush();
        
        current_file_size += full_message.length() + 1; // +1 for newline
        if (current_file_size > max_file_size) {
            rotateLogFile();
        }
    }
}

void CrossoverLogger::logError(const std::string& message, const std::string& function, int line) {
    log(LogLevel::ERROR, message, function, line);
}

void CrossoverLogger::logWarning(const std::string& message, const std::string& function, int line) {
    log(LogLevel::WARNING, message, function, line);
}

void CrossoverLogger::logInfo(const std::string& message, const std::string& function, int line) {
    log(LogLevel::INFO, message, function, line);
}

void CrossoverLogger::logDebug(const std::string& message, const std::string& function, int line) {
    log(LogLevel::DEBUG, message, function, line);
}

CrossoverLogger::~CrossoverLogger() {
    if (log_file.is_open()) {
        log_file.close();
    }
}

// ============================================================================
// BASE CROSSOVER OPERATOR IMPLEMENTATION
// ============================================================================

void CrossoverOperator::logOperation(const std::string& operation, bool success) const {
    const_cast<size_t&>(operation_count)++;
    if (!success) {
        const_cast<size_t&>(error_count)++;
    }
    
    std::stringstream msg;
    msg << operator_name << ": " << operation << " - " << (success ? "SUCCESS" : "FAILED")
        << " (Operations: " << operation_count << ", Errors: " << error_count << ")";
    
    if (success) {
        LOG_DEBUG(msg.str());
    } else {
        LOG_ERROR(msg.str());
    }
}

void CrossoverOperator::logError(const std::string& error_msg) const {
    const_cast<size_t&>(error_count)++;
    std::stringstream msg;
    msg << operator_name << " ERROR: " << error_msg;
    LOG_ERROR(msg.str());
}

// ============================================================================
// TREE NODE IMPLEMENTATION
// ============================================================================

TreeNode::~TreeNode() {
    for (auto* child : children) {
        delete child;
    }
}

TreeNode* TreeNode::clone() const {
    TreeNode* copy = new TreeNode(value);
    for (const auto* child : children) {
        copy->children.push_back(child->clone());
    }
    return copy;
}
