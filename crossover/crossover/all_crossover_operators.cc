#include "all_crossover_operators.h"
#include <algorithm>
#include <stdexcept>

namespace crossover {
    
    std::unique_ptr<CrossoverOperator> CreateCrossoverOperator(const std::string& name) {
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        
        if (lower_name == "onepoint" || lower_name == "one_point" || lower_name == "one-point") {
            return std::make_unique<OnePointCrossover>();
        }
        else if (lower_name == "twopoint" || lower_name == "two_point" || lower_name == "two-point") {
            return std::make_unique<TwoPointCrossover>();
        }
        else if (lower_name == "multipoint" || lower_name == "multi_point" || lower_name == "multi-point") {
            return std::make_unique<MultiPointCrossover>();
        }
        else if (lower_name == "uniform") {
            return std::make_unique<UniformCrossover>();
        }
        else if (lower_name == "uniform_k_vector" || lower_name == "uniform-k-vector") {
            return std::make_unique<UniformKVectorCrossover>();
        }
        else if (lower_name == "blend" || lower_name == "blx" || lower_name == "blx-alpha") {
            return std::make_unique<BlendCrossover>();
        }
        else if (lower_name == "sbx" || lower_name == "simulated_binary" || lower_name == "simulated-binary") {
            return std::make_unique<SimulatedBinaryCrossover>();
        }
        else if (lower_name == "line_recombination" || lower_name == "line-recombination") {
            return std::make_unique<LineRecombination>();
        }
        else if (lower_name == "intermediate" || lower_name == "intermediate_recombination") {
            return std::make_unique<IntermediateRecombination>();
        }
        else if (lower_name == "cut_and_crossfill" || lower_name == "cut-and-crossfill") {
            return std::make_unique<CutAndCrossfillCrossover>();
        }
        else if (lower_name == "pmx" || lower_name == "partially_mapped" || lower_name == "partially-mapped") {
            return std::make_unique<PartiallyMappedCrossover>();
        }
        else if (lower_name == "edge" || lower_name == "edge_crossover") {
            return std::make_unique<EdgeCrossover>();
        }
        else if (lower_name == "order" || lower_name == "ox" || lower_name == "order_crossover") {
            return std::make_unique<OrderCrossover>();
        }
        else if (lower_name == "cycle" || lower_name == "cx" || lower_name == "cycle_crossover") {
            return std::make_unique<CycleCrossover>();
        }
        else if (lower_name == "subtree" || lower_name == "subtree_crossover") {
            return std::make_unique<SubtreeCrossover>();
        }
        else if (lower_name == "diploid" || lower_name == "diploid_recombination") {
            return std::make_unique<DiploidRecombination>();
        }
        else if (lower_name == "differential_evolution" || lower_name == "de") {
            return std::make_unique<DifferentialEvolutionCrossover>();
        }
        else {
            throw std::invalid_argument("Unknown crossover operator: " + name);
        }
    }
    
    std::vector<std::string> GetAvailableCrossoverTypes() {
        return {
            "one_point", "two_point", "multi_point", "uniform", "uniform_k_vector",
            "blend", "simulated_binary", "line_recombination", "intermediate_recombination",
            "cut_and_crossfill", "partially_mapped", "edge", "order", "cycle", 
            "subtree", "diploid", "differential_evolution"
        };
    }
    
    bool IsValidOperator(const std::string& name) {
        try {
            CreateCrossoverOperator(name);
            return true;
        } catch (const std::invalid_argument&) {
            return false;
        }
    }
    
} // namespace crossover
            return std::make_unique<SubtreeCrossover>(seed);
        }
        else if (lower_name == "diploid" || lower_name == "diploid_recombination") {
            return std::make_unique<DiploidRecombination>(seed);
        }
        else if (lower_name == "differential_evolution" || lower_name == "de") {
            return std::make_unique<DifferentialEvolutionCrossover>(0.5, seed);
        }
        else if (lower_name == "uniform_k_vector" || lower_name == "uniform-k-vector") {
            return std::make_unique<UniformKVectorCrossover>(0.1, seed);
        }
        else {
            throw std::invalid_argument("Unknown crossover operator: " + name);
        }
    }
    
    std::vector<std::string> getAvailableOperators() {
        return {
            "OnePoint", "TwoPoint", "MultiPoint", "Uniform", "UniformKVector",
            "Blend", "SimulatedBinary", "LineRecombination", "Intermediate",
            "CutAndCrossfill", "PartiallyMapped", "Edge", "Order", "Cycle",
            "Subtree", "Diploid", "DifferentialEvolution"
        };
    }
    
    bool isValidOperator(const std::string& name) {
        auto operators = getAvailableOperators();
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        
        for (const auto& op : operators) {
            std::string lower_op = op;
            std::transform(lower_op.begin(), lower_op.end(), lower_op.begin(), ::tolower);
            if (lower_name == lower_op) {
                return true;
            }
        }
        
        // Check common aliases
        return (lower_name == "one_point" || lower_name == "one-point" ||
                lower_name == "two_point" || lower_name == "two-point" ||
                lower_name == "multi_point" || lower_name == "multi-point" ||
                lower_name == "blx" || lower_name == "blx-alpha" ||
                lower_name == "sbx" || lower_name == "simulated_binary" ||
                lower_name == "ox" || lower_name == "order_crossover");
    }
}
