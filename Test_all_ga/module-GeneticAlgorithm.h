/* $Header$ */
/*
 * MBDyn (C) is a multibody analysis code.
 * http://www.mbdyn.org
 *
 * Copyright (C) 1996-2023
 *
 * Pierangelo Masarati  <pierangelo.masarati@polimi.it>
 *
 * Dipartimento di Ingegneria Aerospaziale - Politecnico di Milano
 * via La Masa, 34 - 20156 Milano, Italy
 * http://www.aero.polimi.it
 *
 * Changing this copyright notice is forbidden.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation (version 2 of the License).
 * 
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifndef MODULE_GENETIC_ALGORITHM_H
#define MODULE_GENETIC_ALGORITHM_H

#include <vector>
#include <string>
#include <memory>
#include <random>

// Include modularized operators
#include "crossover/all_crossover_operators.h"
#include "mutation/all_mutation_operators.h"
#include "selection-operator/all_selection_operators.h"

namespace GeneticAlgorithm {

/**
 * @brief Individual in the genetic algorithm population
 */
struct Individual {
    std::vector<double> chromosome;
    double fitness;
    bool evaluated;

    Individual() : fitness(0.0), evaluated(false) {}
    Individual(const std::vector<double>& chromo) 
        : chromosome(chromo), fitness(0.0), evaluated(false) {}
};

/**
 * @brief Population of individuals
 */
typedef std::vector<Individual> Population;

/**
 * @brief Genetic Algorithm Parameters
 */
struct GAParameters {
    int population_size;
    int max_generations;
    double crossover_rate;
    double mutation_rate;
    double elitism_rate;
    int tournament_size;
    std::string crossover_type;
    std::string mutation_type;
    std::string selection_type;
    double convergence_threshold;
    int chromosome_length;
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
    
    GAParameters() :
        population_size(100),
        max_generations(1000),
        crossover_rate(0.8),
        mutation_rate(0.1),
        elitism_rate(0.1),
        tournament_size(3),
        crossover_type("one_point"),
        mutation_type("gaussian"),
        selection_type("tournament"),
        convergence_threshold(1e-6),
        chromosome_length(10) {}
};

/**
 * @brief Main Genetic Algorithm class
 */
class GeneticAlgorithmOptimizer {
private:
    GAParameters params;
    Population population;
    std::mt19937 rng;
    std::unique_ptr<crossover::BaseCrossover> crossover_op;
    std::unique_ptr<mutation::BaseMutation> mutation_op;
    std::unique_ptr<selection::BaseSelection> selection_op;
    
    // Best solution tracking
    Individual best_individual;
    std::vector<double> fitness_history;
    
public:
    /**
     * @brief Constructor
     */
    explicit GeneticAlgorithmOptimizer(const GAParameters& parameters);
    
    /**
     * @brief Destructor
     */
    ~GeneticAlgorithmOptimizer();
    
    /**
     * @brief Initialize the population
     */
    void InitializePopulation();
    
    /**
     * @brief Evaluate fitness for an individual
     * @param individual The individual to evaluate
     * @return The fitness value
     */
    virtual double EvaluateFitness(const Individual& individual) = 0;
    
    /**
     * @brief Evaluate the entire population
     */
    void EvaluatePopulation();
    
    /**
     * @brief Run the genetic algorithm
     * @return The best individual found
     */
    Individual Run();
    
    /**
     * @brief Get the current best individual
     */
    const Individual& GetBestIndividual() const { return best_individual; }
    
    /**
     * @brief Get fitness history
     */
    const std::vector<double>& GetFitnessHistory() const { return fitness_history; }
    
    /**
     * @brief Get current population
     */
    const Population& GetPopulation() const { return population; }
    
    /**
     * @brief Set parameters
     */
    void SetParameters(const GAParameters& parameters);
    
    /**
     * @brief Get parameters
     */
    const GAParameters& GetParameters() const { return params; }

protected:
    /**
     * @brief Create operators based on parameter settings
     */
    void CreateOperators();
    
    /**
     * @brief Perform one generation of evolution
     */
    void EvolveGeneration();
    
    /**
     * @brief Update best individual if necessary
     */
    void UpdateBest();
    
    /**
     * @brief Check convergence criteria
     */
    bool HasConverged() const;
    
    /**
     * @brief Apply elitism (preserve best individuals)
     */
    void ApplyElitism(Population& new_population);
};

/**
 * @brief Factory function to create GA optimizer instances
 */
std::unique_ptr<GeneticAlgorithmOptimizer> CreateGAOptimizer(const GAParameters& params);

} // namespace GeneticAlgorithm

#endif // MODULE_GENETIC_ALGORITHM_H
