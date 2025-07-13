#include "all_crossover_operators.h"
#include <iostream>
#include <vector>

/**
 * @brief Simple test program to verify all crossover operators work
 */
int main() {
    std::cout << "=== Testing All Crossover Operators ===" << std::endl;
    
    // Test binary/bit string crossovers
    std::cout << "\n--- Binary Crossover Tests ---" << std::endl;
    BitString parent1_bin = {true, false, true, false, true, false, true, false};
    BitString parent2_bin = {false, true, false, true, false, true, false, true};
    
    std::vector<std::string> binary_ops = {"OnePoint", "TwoPoint", "MultiPoint", "Uniform"};
    for (const auto& op_name : binary_ops) {
        try {
            auto op = CrossoverOperators::create(op_name);
            auto [child1, child2] = op->crossover(parent1_bin, parent2_bin);
            std::cout << op_name << " crossover: SUCCESS" << std::endl;
        } catch (const std::exception& e) {
            std::cout << op_name << " crossover: FAILED - " << e.what() << std::endl;
        }
    }
    
    // Test real-valued crossovers
    std::cout << "\n--- Real-Valued Crossover Tests ---" << std::endl;
    RealVector parent1_real = {1.0, 2.0, 3.0, 4.0, 5.0};
    RealVector parent2_real = {6.0, 7.0, 8.0, 9.0, 10.0};
    
    std::vector<std::string> real_ops = {"OnePoint", "TwoPoint", "MultiPoint", "Uniform", 
                                        "Blend", "SimulatedBinary", "LineRecombination", "Intermediate"};
    for (const auto& op_name : real_ops) {
        try {
            auto op = CrossoverOperators::create(op_name);
            auto [child1, child2] = op->crossover(parent1_real, parent2_real);
            std::cout << op_name << " crossover: SUCCESS" << std::endl;
        } catch (const std::exception& e) {
            std::cout << op_name << " crossover: FAILED - " << e.what() << std::endl;
        }
    }
    
    // Test permutation crossovers
    std::cout << "\n--- Permutation Crossover Tests ---" << std::endl;
    Permutation parent1_perm = {1, 2, 3, 4, 5, 6};
    Permutation parent2_perm = {6, 5, 4, 3, 2, 1};
    
    std::vector<std::string> perm_ops = {"CutAndCrossfill", "PartiallyMapped", "Order", "Cycle"};
    for (const auto& op_name : perm_ops) {
        try {
            auto op = CrossoverOperators::create(op_name);
            auto [child1, child2] = op->crossover(parent1_perm, parent2_perm);
            std::cout << op_name << " crossover: SUCCESS" << std::endl;
        } catch (const std::exception& e) {
            std::cout << op_name << " crossover: FAILED - " << e.what() << std::endl;
        }
    }
    
    // Test edge crossover (single child)
    try {
        auto edge_op = CrossoverOperators::create("Edge");
        auto child = static_cast<EdgeCrossover*>(edge_op.get())->performCrossover(parent1_perm, parent2_perm);
        std::cout << "Edge crossover: SUCCESS" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Edge crossover: FAILED - " << e.what() << std::endl;
    }
    
    // Test differential evolution crossover
    std::cout << "\n--- Differential Evolution Test ---" << std::endl;
    try {
        auto de_op = CrossoverOperators::create("DifferentialEvolution");
        auto trial = static_cast<DifferentialEvolutionCrossover*>(de_op.get())->performCrossover(parent1_real, parent2_real);
        std::cout << "DifferentialEvolution crossover: SUCCESS" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "DifferentialEvolution crossover: FAILED - " << e.what() << std::endl;
    }
    
    // Test diploid recombination
    std::cout << "\n--- Diploid Recombination Test ---" << std::endl;
    try {
        auto diploid_op = CrossoverOperators::create("Diploid");
        DiploidRecombination::DiploidChromosome diploid_parent1 = {parent1_bin, parent2_bin};
        DiploidRecombination::DiploidChromosome diploid_parent2 = {parent2_bin, parent1_bin};
        auto offspring = static_cast<DiploidRecombination*>(diploid_op.get())->crossover(diploid_parent1, diploid_parent2);
        std::cout << "Diploid crossover: SUCCESS" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Diploid crossover: FAILED - " << e.what() << std::endl;
    }
    
    std::cout << "\n=== All Tests Completed ===" << std::endl;
    std::cout << "Available operators: ";
    auto available = CrossoverOperators::getAvailableOperators();
    for (size_t i = 0; i < available.size(); ++i) {
        std::cout << available[i];
        if (i < available.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    
    return 0;
}
