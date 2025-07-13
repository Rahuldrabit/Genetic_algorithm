#include "all_selection_operators.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>

namespace SelectionOperators {

std::unique_ptr<SelectionOperator> createSelectionOperator(SelectionType type, 
                                                          const std::map<std::string, double>& params) {
    switch (type) {
        case SelectionType::TOURNAMENT: {
            size_t tournament_size = 3; // default
            auto it = params.find("tournament_size");
            if (it != params.end() && it->second > 0) {
                tournament_size = static_cast<size_t>(it->second);
            }
            return std::make_unique<TournamentSelection>(tournament_size);
        }
        
        case SelectionType::ROULETTE_WHEEL:
            return std::make_unique<RouletteWheelSelection>();
            
        case SelectionType::RANK: {
            double pressure = 2.0; // default
            auto it = params.find("selection_pressure");
            if (it != params.end() && it->second >= 1.0) {
                pressure = it->second;
            }
            return std::make_unique<RankSelection>(pressure);
        }
        
        case SelectionType::STOCHASTIC_UNIVERSAL_SAMPLING:
            return std::make_unique<StochasticUniversalSampling>();
            
        case SelectionType::ELITISM:
            return std::make_unique<ElitismSelection>();
            
        default:
            return nullptr;
    }
}

std::vector<std::string> getAvailableOperators() {
    return {
        "Tournament",
        "RouletteWheel", 
        "Rank",
        "StochasticUniversalSampling",
        "Elitism"
    };
}

std::unique_ptr<SelectionOperator> createByName(const std::string& name,
                                               const std::map<std::string, double>& params) {
    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    
    if (lower_name == "tournament") {
        return createSelectionOperator(SelectionType::TOURNAMENT, params);
    } else if (lower_name == "roulettewheel" || lower_name == "roulette_wheel" || lower_name == "roulette") {
        return createSelectionOperator(SelectionType::ROULETTE_WHEEL, params);
    } else if (lower_name == "rank") {
        return createSelectionOperator(SelectionType::RANK, params);
    } else if (lower_name == "stochasticuniversalsampling" || lower_name == "stochastic_universal_sampling" || lower_name == "sus") {
        return createSelectionOperator(SelectionType::STOCHASTIC_UNIVERSAL_SAMPLING, params);
    } else if (lower_name == "elitism" || lower_name == "elite") {
        return createSelectionOperator(SelectionType::ELITISM, params);
    }
    
    return nullptr;
}

void demonstrateAllOperators(const std::vector<Individual>& population, size_t select_count) {
    if (population.empty()) {
        std::cout << "Cannot demonstrate with empty population" << std::endl;
        return;
    }
    
    std::cout << "=== Selection Operators Demonstration ===" << std::endl;
    std::cout << "Population size: " << population.size() << std::endl;
    std::cout << "Selecting: " << select_count << " individuals" << std::endl;
    std::cout << std::endl;
    
    auto operators = getAvailableOperators();
    
    for (const auto& op_name : operators) {
        std::cout << "--- " << op_name << " Selection ---" << std::endl;
        
        try {
            auto op = createByName(op_name);
            if (op) {
                auto start = std::chrono::high_resolution_clock::now();
                auto selected = op->select(population, select_count);
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                
                std::cout << "Selected " << selected.size() << " individuals" << std::endl;
                std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;
                
                // Show fitness of selected individuals
                std::cout << "Selected fitness values: ";
                for (size_t i = 0; i < std::min(selected.size(), size_t(5)); ++i) {
                    std::cout << selected[i].fitness << " ";
                }
                if (selected.size() > 5) {
                    std::cout << "...";
                }
                std::cout << std::endl;
                
                std::cout << "Operations performed: " << op->getOperationCount() << std::endl;
                std::cout << "SUCCESS" << std::endl;
            } else {
                std::cout << "FAILED: Could not create operator" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << std::endl;
        }
        
        std::cout << std::endl;
    }
}

void compareOperatorPerformance(const std::vector<Individual>& population,
                               size_t select_count,
                               size_t iterations) {
    if (population.empty()) {
        std::cout << "Cannot compare with empty population" << std::endl;
        return;
    }
    
    std::cout << "=== Selection Operator Performance Comparison ===" << std::endl;
    std::cout << "Population size: " << population.size() << std::endl;
    std::cout << "Select count: " << select_count << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << std::endl;
    
    auto operators = getAvailableOperators();
    
    for (const auto& op_name : operators) {
        auto op = createByName(op_name);
        if (!op) continue;
        
        std::cout << "Testing " << op_name << "..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < iterations; ++i) {
            auto selected = op->select(population, select_count);
            // Use the result to prevent optimization
            volatile size_t result_size = selected.size();
            (void)result_size;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double avg_time = static_cast<double>(total_duration.count()) / iterations;
        
        std::cout << "  Average time: " << avg_time << " microseconds" << std::endl;
        std::cout << "  Total operations: " << op->getOperationCount() << std::endl;
        std::cout << std::endl;
    }
}

} // namespace SelectionOperators
