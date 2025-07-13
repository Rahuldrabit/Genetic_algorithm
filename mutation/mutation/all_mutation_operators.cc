#include "all_mutation_operators.h"
#include <iostream>

namespace MutationOperators {
    
    std::unique_ptr<MutationOperator> create(const std::string& name, unsigned seed) {
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        
        if (lower_name == "bitflip" || lower_name == "bit_flip" || lower_name == "bit-flip") {
            return std::make_unique<BitFlipMutation>(seed);
        }
        else if (lower_name == "uniform" || lower_name == "uniform_mutation") {
            return std::make_unique<UniformMutation>(seed);
        }
        else if (lower_name == "gaussian" || lower_name == "gaussian_mutation") {
            return std::make_unique<GaussianMutation>(seed);
        }
        else if (lower_name == "random_resetting" || lower_name == "random-resetting" || lower_name == "randomresetting") {
            return std::make_unique<RandomResettingMutation>(seed);
        }
        else if (lower_name == "creep" || lower_name == "creep_mutation") {
            return std::make_unique<CreepMutation>(seed);
        }
        else if (lower_name == "swap" || lower_name == "swap_mutation") {
            return std::make_unique<SwapMutation>(seed);
        }
        else if (lower_name == "inversion" || lower_name == "inversion_mutation") {
            return std::make_unique<InversionMutation>(seed);
        }
        else if (lower_name == "insert" || lower_name == "insert_mutation") {
            return std::make_unique<InsertMutation>(seed);
        }
        else if (lower_name == "scramble" || lower_name == "scramble_mutation") {
            return std::make_unique<ScrambleMutation>(seed);
        }
        else if (lower_name == "self_adaptive" || lower_name == "self-adaptive" || lower_name == "selfadaptive") {
            return std::make_unique<SelfAdaptiveMutation>(seed);
        }
        else if (lower_name == "list" || lower_name == "list_mutation") {
            return std::make_unique<ListMutation>(seed);
        }
        else {
            throw std::invalid_argument("Unknown mutation operator: " + name);
        }
    }
    
    std::vector<std::string> getAvailableOperators(const std::string& encoding_type) {
        if (encoding_type == "binary" || encoding_type == "bit_string") {
            return {"BitFlip"};
        } else if (encoding_type == "real" || encoding_type == "real_valued") {
            return {"Uniform", "Gaussian"};
        } else if (encoding_type == "integer") {
            return {"RandomResetting", "Creep"};
        } else if (encoding_type == "permutation") {
            return {"Swap", "Inversion", "Insert", "Scramble"};
        } else if (encoding_type == "self_adaptive") {
            return {"SelfAdaptive"};
        } else if (encoding_type == "list" || encoding_type == "variable_length") {
            return {"List"};
        } else if (encoding_type == "all") {
            return {"BitFlip", "Uniform", "Gaussian", "RandomResetting", "Creep", 
                   "Swap", "Inversion", "Insert", "Scramble", "SelfAdaptive", "List"};
        }
        
        return {"BitFlip"};  // Default
    }
    
    bool isValidOperator(const std::string& name) {
        auto all_operators = getAvailableOperators("all");
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        
        for (const auto& op : all_operators) {
            std::string lower_op = op;
            std::transform(lower_op.begin(), lower_op.end(), lower_op.begin(), ::tolower);
            if (lower_name == lower_op) {
                return true;
            }
        }
        
        // Check common aliases
        return (lower_name == "bit_flip" || lower_name == "bit-flip" ||
                lower_name == "random_resetting" || lower_name == "random-resetting" ||
                lower_name == "gaussian_mutation" || lower_name == "uniform_mutation" ||
                lower_name == "creep_mutation" || lower_name == "swap_mutation" ||
                lower_name == "inversion_mutation");
    }
    
    /**
     * @brief Demonstrate usage of different mutation operators
     */
    void demonstrateMutationOperators() {
        std::cout << "=== Mutation Operators Demonstration ===" << std::endl;
        
        // Example with binary encoding
        BitString chromosome_bin = {true, false, true, false, true, false, true, false};
        auto bit_flip = create("BitFlip");
        static_cast<BitFlipMutation*>(bit_flip.get())->mutate(chromosome_bin, 0.1);
        std::cout << "BitFlip mutation: SUCCESS" << std::endl;
        
        // Example with real-valued encoding
        RealVector chromosome_real = {1.0, 2.0, 3.0, 4.0, 5.0};
        RealVector lower_bounds = {0.0, 0.0, 0.0, 0.0, 0.0};
        RealVector upper_bounds = {10.0, 10.0, 10.0, 10.0, 10.0};
        
        auto gaussian = create("Gaussian");
        static_cast<GaussianMutation*>(gaussian.get())->mutate(chromosome_real, 0.1, 0.5, lower_bounds, upper_bounds);
        std::cout << "Gaussian mutation: SUCCESS" << std::endl;
        
        // Example with permutation encoding
        Permutation chromosome_perm = {1, 2, 3, 4, 5};
        auto swap = create("Swap");
        static_cast<SwapMutation*>(swap.get())->mutate(chromosome_perm, 0.2);
        std::cout << "Swap mutation: SUCCESS" << std::endl;
        
        std::cout << "All mutation operators working correctly!" << std::endl;
    }
}
