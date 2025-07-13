/* $Header$ */
/*
 * Example usage of the modularized Genetic Algorithm operators
 * 
 * This file demonstrates how to use all the crossover, mutation, and selection
 * operators that have been modularized in the genetic algorithm codebase.
 */

#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <cmath>

#include "module-GeneticAlgorithm.h"

using namespace GeneticAlgorithm;

/**
 * @brief Example optimization problem: minimize the sphere function
 * f(x) = sum(xi^2) for i=1 to n
 */
class SphereOptimizer : public GeneticAlgorithmOptimizer {
public:
    explicit SphereOptimizer(const GAParameters& params) 
        : GeneticAlgorithmOptimizer(params) {}
    
    double EvaluateFitness(const Individual& individual) override {
        double sum = 0.0;
        for (double gene : individual.chromosome) {
            sum += gene * gene;
        }
        // Return negative because GA maximizes fitness, but we want to minimize
        return -sum;
    }
};

/**
 * @brief Example optimization problem: maximize the Rastrigin function
 */
class RastriginOptimizer : public GeneticAlgorithmOptimizer {
private:
    static constexpr double A = 10.0;
    
public:
    explicit RastriginOptimizer(const GAParameters& params) 
        : GeneticAlgorithmOptimizer(params) {}
    
    double EvaluateFitness(const Individual& individual) override {
        double sum = A * individual.chromosome.size();
        for (double gene : individual.chromosome) {
            sum += gene * gene - A * std::cos(2.0 * M_PI * gene);
        }
        // Return negative to maximize (minimize the original function)
        return -sum;
    }
};

void TestAllCrossoverOperators() {
    std::cout << "\n=== Testing All Crossover Operators ===" << std::endl;
    
    std::mt19937 rng(42);
    
    // Create test individuals
    Individual parent1, parent2;
    parent1.chromosome = {1.0, 2.0, 3.0, 4.0, 5.0};
    parent2.chromosome = {6.0, 7.0, 8.0, 9.0, 10.0};
    
    std::vector<std::string> crossover_types = {
        "one_point", "two_point", "multi_point", "uniform", 
        "uniform_k_vector", "blend", "simulated_binary", 
        "line_recombination", "intermediate_recombination"
    };
    
    for (const auto& type : crossover_types) {
        std::cout << "\nTesting " << type << " crossover:" << std::endl;
        
        auto crossover_op = crossover::CreateCrossoverOperator(type);
        if (crossover_op) {
            auto offspring = crossover_op->Crossover(parent1, parent2, rng);
            
            std::cout << "  Offspring 1: ";
            for (double gene : offspring[0].chromosome) {
                std::cout << gene << " ";
            }
            std::cout << std::endl;
            
            std::cout << "  Offspring 2: ";
            for (double gene : offspring[1].chromosome) {
                std::cout << gene << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "  Failed to create operator!" << std::endl;
        }
    }
}

void TestAllMutationOperators() {
    std::cout << "\n=== Testing All Mutation Operators ===" << std::endl;
    
    std::mt19937 rng(42);
    
    // Create test individual
    Individual individual;
    individual.chromosome = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    std::vector<std::string> mutation_types = {
        "bit_flip", "random_resetting", "creep", "uniform", 
        "gaussian", "swap", "inversion", "insert", "scramble"
    };
    
    for (const auto& type : mutation_types) {
        std::cout << "\nTesting " << type << " mutation:" << std::endl;
        
        auto mutation_op = mutation::CreateMutationOperator(type);
        if (mutation_op) {
            Individual original = individual;
            Individual mutated = mutation_op->Mutate(individual, rng);
            
            std::cout << "  Original:  ";
            for (double gene : original.chromosome) {
                std::cout << gene << " ";
            }
            std::cout << std::endl;
            
            std::cout << "  Mutated:   ";
            for (double gene : mutated.chromosome) {
                std::cout << gene << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "  Failed to create operator!" << std::endl;
        }
    }
}

void TestAllSelectionOperators() {
    std::cout << "\n=== Testing All Selection Operators ===" << std::endl;
    
    std::mt19937 rng(42);
    
    // Create test population
    Population population;
    for (int i = 0; i < 10; ++i) {
        Individual ind;
        ind.chromosome = {static_cast<double>(i)};
        ind.fitness = i * i; // Quadratic fitness
        ind.evaluated = true;
        population.push_back(ind);
    }
    
    std::vector<std::string> selection_types = {
        "tournament", "roulette_wheel", "rank", 
        "stochastic_universal_sampling", "elitism"
    };
    
    for (const auto& type : selection_types) {
        std::cout << "\nTesting " << type << " selection:" << std::endl;
        
        auto selection_op = selection::CreateSelectionOperator(type);
        if (selection_op) {
            std::cout << "  Selected indices: ";
            for (int i = 0; i < 5; ++i) {
                int selected = selection_op->Select(population, rng);
                std::cout << selected << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "  Failed to create operator!" << std::endl;
        }
    }
}

void RunFullGAExample() {
    std::cout << "\n=== Running Full GA Example ===" << std::endl;
    
    // Set up parameters for sphere function optimization
    GAParameters params;
    params.population_size = 50;
    params.max_generations = 100;
    params.crossover_rate = 0.8;
    params.mutation_rate = 0.1;
    params.elitism_rate = 0.1;
    params.tournament_size = 3;
    params.crossover_type = "simulated_binary";
    params.mutation_type = "gaussian";
    params.selection_type = "tournament";
    params.convergence_threshold = 1e-6;
    params.chromosome_length = 5;
    params.lower_bounds = std::vector<double>(5, -10.0);
    params.upper_bounds = std::vector<double>(5, 10.0);
    
    std::cout << "Optimizing 5D sphere function..." << std::endl;
    SphereOptimizer sphere_optimizer(params);
    Individual best_sphere = sphere_optimizer.Run();
    
    std::cout << "Best solution found:" << std::endl;
    std::cout << "  Fitness: " << best_sphere.fitness << std::endl;
    std::cout << "  Chromosome: ";
    for (double gene : best_sphere.chromosome) {
        std::cout << gene << " ";
    }
    std::cout << std::endl;
    
    // Test with different parameters for Rastrigin function
    params.crossover_type = "blend";
    params.mutation_type = "uniform";
    params.selection_type = "rank";
    params.max_generations = 200;
    
    std::cout << "\nOptimizing 5D Rastrigin function..." << std::endl;
    RastriginOptimizer rastrigin_optimizer(params);
    Individual best_rastrigin = rastrigin_optimizer.Run();
    
    std::cout << "Best solution found:" << std::endl;
    std::cout << "  Fitness: " << best_rastrigin.fitness << std::endl;
    std::cout << "  Chromosome: ";
    for (double gene : best_rastrigin.chromosome) {
        std::cout << gene << " ";
    }
    std::cout << std::endl;
}

void TestOperatorFactories() {
    std::cout << "\n=== Testing Operator Factories ===" << std::endl;
    
    // Test crossover factory
    std::cout << "\nAvailable crossover operators:" << std::endl;
    auto crossover_types = crossover::GetAvailableCrossoverTypes();
    for (const auto& type : crossover_types) {
        std::cout << "  - " << type << std::endl;
    }
    
    // Test mutation factory
    std::cout << "\nAvailable mutation operators:" << std::endl;
    auto mutation_types = mutation::GetAvailableMutationTypes();
    for (const auto& type : mutation_types) {
        std::cout << "  - " << type << std::endl;
    }
    
    // Test selection factory
    std::cout << "\nAvailable selection operators:" << std::endl;
    auto selection_types = selection::GetAvailableSelectionTypes();
    for (const auto& type : selection_types) {
        std::cout << "  - " << type << std::endl;
    }
}

int main() {
    std::cout << "Genetic Algorithm Modularized Operators Test" << std::endl;
    std::cout << "============================================" << std::endl;
    
    try {
        TestOperatorFactories();
        TestAllCrossoverOperators();
        TestAllMutationOperators();  
        TestAllSelectionOperators();
        RunFullGAExample();
        
        std::cout << "\n=== All Tests Completed Successfully ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
