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

#include "mbconfig.h"           /* This goes first in every *.c,*.cc file */

#include <iostream>
#include <algorithm>
#include <numeric>
#include <cfloat>
#include <cmath>

#include "module-GeneticAlgorithm.h"
#include <dataman.h>
#include "loadable.h"

namespace GeneticAlgorithm {

GeneticAlgorithmOptimizer::GeneticAlgorithmOptimizer(const GAParameters& parameters)
    : params(parameters), rng(std::random_device{}())
{
    CreateOperators();
}

GeneticAlgorithmOptimizer::~GeneticAlgorithmOptimizer() = default;

void GeneticAlgorithmOptimizer::CreateOperators() {
    // Create crossover operator
    crossover_op = crossover::CreateCrossoverOperator(params.crossover_type);
    
    // Create mutation operator  
    mutation_op = mutation::CreateMutationOperator(params.mutation_type);
    
    // Create selection operator
    selection_op = selection::CreateSelectionOperator(params.selection_type);
}

void GeneticAlgorithmOptimizer::InitializePopulation() {
    population.clear();
    population.reserve(params.population_size);
    
    std::uniform_real_distribution<double> dist;
    
    for (int i = 0; i < params.population_size; ++i) {
        Individual individual;
        individual.chromosome.resize(params.chromosome_length);
        
        for (int j = 0; j < params.chromosome_length; ++j) {
            if (j < static_cast<int>(params.lower_bounds.size()) && 
                j < static_cast<int>(params.upper_bounds.size())) {
                std::uniform_real_distribution<double> gene_dist(
                    params.lower_bounds[j], params.upper_bounds[j]);
                individual.chromosome[j] = gene_dist(rng);
            } else {
                individual.chromosome[j] = dist(rng);
            }
        }
        
        population.push_back(individual);
    }
}

void GeneticAlgorithmOptimizer::EvaluatePopulation() {
    for (auto& individual : population) {
        if (!individual.evaluated) {
            individual.fitness = EvaluateFitness(individual);
            individual.evaluated = true;
        }
    }
}

Individual GeneticAlgorithmOptimizer::Run() {
    InitializePopulation();
    EvaluatePopulation();
    UpdateBest();
    
    for (int generation = 0; generation < params.max_generations; ++generation) {
        EvolveGeneration();
        EvaluatePopulation();
        UpdateBest();
        
        // Record fitness history
        fitness_history.push_back(best_individual.fitness);
        
        // Check convergence
        if (HasConverged()) {
            std::cout << "Converged at generation " << generation << std::endl;
            break;
        }
        
        // Optional: Print progress
        if (generation % 100 == 0) {
            std::cout << "Generation " << generation 
                      << ", Best fitness: " << best_individual.fitness << std::endl;
        }
    }
    
    return best_individual;
}

void GeneticAlgorithmOptimizer::EvolveGeneration() {
    Population new_population;
    new_population.reserve(params.population_size);
    
    // Apply elitism - preserve best individuals
    int num_elite = static_cast<int>(params.elitism_rate * params.population_size);
    if (num_elite > 0) {
        // Sort population by fitness (assuming higher is better)
        std::sort(population.begin(), population.end(), 
                  [](const Individual& a, const Individual& b) {
                      return a.fitness > b.fitness;
                  });
        
        for (int i = 0; i < num_elite; ++i) {
            new_population.push_back(population[i]);
        }
    }
    
    // Generate offspring through crossover and mutation
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    
    while (new_population.size() < static_cast<size_t>(params.population_size)) {
        // Select parents
        std::vector<Individual> parents;
        for (int i = 0; i < 2; ++i) {
            int parent_idx = selection_op->Select(population, rng);
            parents.push_back(population[parent_idx]);
        }
        
        // Apply crossover
        std::vector<Individual> offspring = parents;
        if (prob_dist(rng) < params.crossover_rate) {
            auto crossed = crossover_op->Crossover(parents[0], parents[1], rng);
            offspring = crossed;
        }
        
        // Apply mutation
        for (auto& child : offspring) {
            if (prob_dist(rng) < params.mutation_rate) {
                child = mutation_op->Mutate(child, rng);
            }
            child.evaluated = false; // Mark for re-evaluation
        }
        
        // Add offspring to new population
        for (auto& child : offspring) {
            if (new_population.size() < static_cast<size_t>(params.population_size)) {
                new_population.push_back(child);
            }
        }
    }
    
    population = new_population;
}

void GeneticAlgorithmOptimizer::UpdateBest() {
    auto best_it = std::max_element(population.begin(), population.end(),
                                    [](const Individual& a, const Individual& b) {
                                        return a.fitness < b.fitness;
                                    });
    
    if (best_it != population.end()) {
        if (best_individual.chromosome.empty() || best_it->fitness > best_individual.fitness) {
            best_individual = *best_it;
        }
    }
}

bool GeneticAlgorithmOptimizer::HasConverged() const {
    if (fitness_history.size() < 10) {
        return false;
    }
    
    // Check if fitness hasn't improved significantly in last 10 generations
    double recent_best = *std::max_element(fitness_history.end() - 10, fitness_history.end());
    double old_best = fitness_history[fitness_history.size() - 10];
    
    return std::abs(recent_best - old_best) < params.convergence_threshold;
}

void GeneticAlgorithmOptimizer::ApplyElitism(Population& new_population) {
    int num_elite = static_cast<int>(params.elitism_rate * params.population_size);
    if (num_elite == 0) return;
    
    // Sort current population by fitness
    std::sort(population.begin(), population.end(),
              [](const Individual& a, const Individual& b) {
                  return a.fitness > b.fitness;
              });
    
    // Replace worst individuals in new population with elite
    std::sort(new_population.begin(), new_population.end(),
              [](const Individual& a, const Individual& b) {
                  return a.fitness > b.fitness;
              });
    
    for (int i = 0; i < num_elite; ++i) {
        if (i < static_cast<int>(new_population.size())) {
            new_population[new_population.size() - 1 - i] = population[i];
        }
    }
}

void GeneticAlgorithmOptimizer::SetParameters(const GAParameters& parameters) {
    params = parameters;
    CreateOperators();
}

} // namespace GeneticAlgorithm

/* MBDyn loadable module interface */

struct module_genetic_algorithm {
    GeneticAlgorithm::GAParameters params;
    std::unique_ptr<GeneticAlgorithm::GeneticAlgorithmOptimizer> optimizer;
    std::vector<double> current_solution;
    double current_fitness;
};

/* Default MBDyn module functions */
static void *
read(LoadableElem* pEl, DataManager* pDM, MBDynParser& HP)
{
    DEBUGCOUTFNAME("genetic_algorithm::read");
    
    auto* p = new module_genetic_algorithm;
    
    // Read parameters from input file
    if (HP.IsKeyWord("population_size")) {
        p->params.population_size = HP.GetInt();
    }
    
    if (HP.IsKeyWord("max_generations")) {
        p->params.max_generations = HP.GetInt();
    }
    
    if (HP.IsKeyWord("crossover_rate")) {
        p->params.crossover_rate = HP.GetReal();
    }
    
    if (HP.IsKeyWord("mutation_rate")) {
        p->params.mutation_rate = HP.GetReal();
    }
    
    if (HP.IsKeyWord("crossover_type")) {
        p->params.crossover_type = HP.GetStringWithDelims();
    }
    
    if (HP.IsKeyWord("mutation_type")) {
        p->params.mutation_type = HP.GetStringWithDelims();
    }
    
    if (HP.IsKeyWord("selection_type")) {
        p->params.selection_type = HP.GetStringWithDelims();
    }
    
    return p;
}

static unsigned int
i_get_num_dof(const LoadableElem* /* pEl */)
{
    DEBUGCOUTFNAME("genetic_algorithm::i_get_num_dof");
    return 0;
}

static std::ostream&
restart(const LoadableElem* pEl, std::ostream& out)
{
    DEBUGCOUTFNAME("genetic_algorithm::restart");
    return out << "# genetic algorithm: not implemented yet" << std::endl;
}

static void
output(const LoadableElem* pEl, OutputHandler& OH)
{
    DEBUGCOUTFNAME("genetic_algorithm::output");
    
    if (OH.UseText(OutputHandler::LOADABLE)) {
        std::ostream& out = OH.Loadable();
        
        module_genetic_algorithm* p = (module_genetic_algorithm*)pEl->pGetData();
        
        out << std::setw(8) << pEl->GetLabel()
            << " " << p->current_fitness;
            
        for (double gene : p->current_solution) {
            out << " " << gene;
        }
        out << std::endl;
    }
}

static void
set_dof(const LoadableElem* /* pEl */, unsigned int /* i */, const DofOwner* /* pDO */)
{
    DEBUGCOUTFNAME("genetic_algorithm::set_dof");
}

static void
set_eq(const LoadableElem* /* pEl */, unsigned int /* i */, const DofOwner* /* pDO */)
{
    DEBUGCOUTFNAME("genetic_algorithm::set_eq");
}

static void
work_space_dim(const LoadableElem* /* pEl */, integer* piNumRows, integer* piNumCols)
{
    DEBUGCOUTFNAME("genetic_algorithm::work_space_dim");
    *piNumRows = 0;
    *piNumCols = 0;
}

static VariableSubMatrixHandler& 
ass_jac(LoadableElem* /* pEl */, 
        VariableSubMatrixHandler& WorkMat,
        doublereal /* dCoef */, 
        const VectorHandler& /* XCurr */,
        const VectorHandler& /* XPrimeCurr */)
{  
    DEBUGCOUTFNAME("genetic_algorithm::ass_jac");
    WorkMat.SetNullMatrix();
    return WorkMat;
}

static void
ass_mats(LoadableElem* /* pEl */, 
         VariableSubMatrixHandler& WorkMatA,
         VariableSubMatrixHandler& WorkMatB,
         const VectorHandler& /* XCurr */,
         const VectorHandler& /* XPrimeCurr */)
{  
    DEBUGCOUTFNAME("genetic_algorithm::ass_mats");
    WorkMatA.SetNullMatrix();
    WorkMatB.SetNullMatrix();
}

static SubVectorHandler& 
ass_res(LoadableElem* /* pEl */, 
        SubVectorHandler& WorkVec,
        doublereal /* dCoef */,
        const VectorHandler& /* XCurr */, 
        const VectorHandler& /* XPrimeCurr */)
{
    DEBUGCOUTFNAME("genetic_algorithm::ass_res");
    WorkVec.Resize(0);
    return WorkVec;
}

static void
before_predict(const LoadableElem* /* pEl */, 
               VectorHandler& /* X */,
               VectorHandler& /* XP */,
               VectorHandler& /* XPrev */,
               VectorHandler& /* XPPrev */)
{
    DEBUGCOUTFNAME("genetic_algorithm::before_predict");
}

static void
after_predict(const LoadableElem* /* pEl */, 
              VectorHandler& /* X */,
              VectorHandler& /* XP */)
{
    DEBUGCOUTFNAME("genetic_algorithm::after_predict");
}

static void
update(LoadableElem* pEl, 
       const VectorHandler& /* X */,
       const VectorHandler& /* XP */)
{
    DEBUGCOUTFNAME("genetic_algorithm::update");
    
    module_genetic_algorithm* p = (module_genetic_algorithm*)pEl->pGetData();
    
    // Run one generation of GA optimization here if needed
    // This is where the main GA logic would be triggered during simulation
}

static void 
after_convergence(const LoadableElem* /* pEl */,
                  const VectorHandler& /* X */,
                  const VectorHandler& /* XP */)
{
    DEBUGCOUTFNAME("genetic_algorithm::after_convergence");
}

static unsigned int
i_get_initial_num_dof(const LoadableElem* /* pEl */)
{
    DEBUGCOUTFNAME("genetic_algorithm::i_get_initial_num_dof");
    return 0;
}

static void
initial_work_space_dim(const LoadableElem* /* pEl */, 
                       integer* piNumRows, 
                       integer* piNumCols)
{
    DEBUGCOUTFNAME("genetic_algorithm::initial_work_space_dim");
    *piNumRows = 0;
    *piNumCols = 0;   
}

static VariableSubMatrixHandler& 
initial_ass_jac(LoadableElem* /* pEl */, 
                 VariableSubMatrixHandler& WorkMat, 
                 const VectorHandler& /* XCurr */)
{
    DEBUGCOUTFNAME("genetic_algorithm::initial_ass_jac");
    WorkMat.SetNullMatrix();
    return WorkMat;
}

static SubVectorHandler& 
initial_ass_res(LoadableElem* /* pEl */, 
                 SubVectorHandler& WorkVec, 
                 const VectorHandler& /* XCurr */)
{
    DEBUGCOUTFNAME("genetic_algorithm::initial_ass_res");
    WorkVec.Resize(0);
    return WorkVec;
}

static void
set_value(const LoadableElem* /* pEl */, DataManager* /* pDM */,
          VectorHandler& /* X */, VectorHandler& /* XP */,
          SimulationEntity::Hints* /* ph */)
{
    DEBUGCOUTFNAME("genetic_algorithm::set_value");
}
   
static void
set_initial_value(const LoadableElem* /* pEl */, VectorHandler& /* X */)
{
    DEBUGCOUTFNAME("genetic_algorithm::set_initial_value");
}

static unsigned int
i_get_num_priv_data(const LoadableElem* /* pEl */)
{
    DEBUGCOUTFNAME("genetic_algorithm::i_get_num_priv_data");
    return 1; // Return fitness value
}

static unsigned int
i_get_priv_data_idx(const LoadableElem* pEl, const char* s)
{
    DEBUGCOUTFNAME("genetic_algorithm::i_get_priv_data_idx");
    
    if (strcmp(s, "fitness") == 0) {
        return 1;
    }
    
    silent_cerr("GeneticAlgorithm Elem(" << pEl->GetLabel() << "): "
        "priv data \"" << s << "\" is unknown" << std::endl);
    throw ErrGeneric(MBDYN_EXCEPT_ARGS);
}

static doublereal
d_get_priv_data(const LoadableElem* pEl, unsigned int i)
{
    DEBUGCOUTFNAME("genetic_algorithm::d_get_priv_data");
    
    module_genetic_algorithm* p = (module_genetic_algorithm*)pEl->pGetData();
    
    switch (i) {
        case 1: // fitness
            return p->current_fitness;
        default:
            silent_cerr("GeneticAlgorithm Elem(" << pEl->GetLabel() << "): "
                "illegal private data index " << i << std::endl);
            throw ErrGeneric(MBDYN_EXCEPT_ARGS);
    }
}

static void
destroy(LoadableElem* pEl)
{
    DEBUGCOUTFNAME("genetic_algorithm::destroy");

    module_genetic_algorithm* p = (module_genetic_algorithm*)pEl->pGetData();
    SAFEDELETE(p);
}

static int
i_get_num_connected_nodes(const LoadableElem* /* pEl */)
{
    DEBUGCOUTFNAME("genetic_algorithm::i_get_num_connected_nodes");
    return 0;
}

static void
get_connected_nodes(const LoadableElem* /* pEl */, 
                    std::vector<const Node*>& connectedNodes)
{
    DEBUGCOUTFNAME("genetic_algorithm::get_connected_nodes");
    connectedNodes.resize(0);
}

static struct LoadableCalls lc = {
    LOADABLE_VERSION_SET(1, 7, 0),

    "genetic_algorithm",
    "1.0",
    "Dipartimento di Ingegneria Aerospaziale, Politecnico di Milano",
    "Genetic Algorithm optimization module with modular crossover, mutation, and selection operators",

    read,
    i_get_num_dof,
    set_dof,
    set_eq,
    output,
    restart,
    work_space_dim,
    ass_jac,
    ass_mats,
    ass_res,
    before_predict,
    after_predict,
    update,
    after_convergence,
    i_get_initial_num_dof,
    initial_work_space_dim,
    initial_ass_jac,
    initial_ass_res,
    set_value,
    set_initial_value,
    i_get_num_priv_data,
    i_get_priv_data_idx,
    d_get_priv_data,
    i_get_num_connected_nodes,
    get_connected_nodes,
    destroy
};

extern "C" {
void* calls = &lc;
}
