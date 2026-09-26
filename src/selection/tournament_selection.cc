#include "tournament_selection.h"
#include <algorithm>
#include <stdexcept>

// ============================================================================
// TOURNAMENT SELECTION IMPLEMENTATION
// ============================================================================

std::vector<Individual> TournamentSelection::select(const std::vector<Individual>& population, size_t count) {
    if (population.empty()) {
        throw std::invalid_argument("Population cannot be empty");
    }
    
    operation_count++;
    
    std::vector<Individual> selected;
    selected.reserve(count);
    
    std::uniform_int_distribution<size_t> dist(0, population.size() - 1);
    
    for (size_t i = 0; i < count; ++i) {
        // Select tournament_size random individuals
        std::vector<size_t> tournament_indices;
        tournament_indices.reserve(tournament_size);
        
        for (size_t j = 0; j < tournament_size && j < population.size(); ++j) {
            tournament_indices.push_back(dist(rng));
        }
        
        // Find the best individual in the tournament
        size_t best_index = tournament_indices[0];
        for (size_t j = 1; j < tournament_indices.size(); ++j) {
            if (population[tournament_indices[j]].fitness > population[best_index].fitness) {
                best_index = tournament_indices[j];
            }
        }
        
        selected.push_back(population[best_index]);
    }
    
    return selected;
}

// Standalone function for compatibility
std::vector<unsigned int> TournamentSelection::selectIndices(std::vector<Individual>& Population, 
                                                            unsigned int TournamentSize) {
    std::vector<unsigned int> SelectedIndices;
    unsigned int PopulationSize = Population.size();

    if (PopulationSize == 0) {
        throw std::invalid_argument("Population is empty");
    }
    if (TournamentSize == 0) {
        throw std::invalid_argument("Tournament size must be greater than 0");
    }
    
    // Adjust tournament size if necessary
    if (TournamentSize > PopulationSize) {
        TournamentSize = PopulationSize;
    }
    
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<unsigned int> dist(0, PopulationSize - 1);
    
    // Randomly select individuals for the tournament
    SelectedIndices.reserve(TournamentSize);
    while (SelectedIndices.size() < TournamentSize) {
        SelectedIndices.push_back(dist(rng));
    }

    // Find the best individual among the selected ones
    unsigned int BestIndex = SelectedIndices[0];
    for (unsigned int i = 1; i < SelectedIndices.size(); ++i) {
        if (Population[SelectedIndices[i]].fitness > Population[BestIndex].fitness) {
            BestIndex = SelectedIndices[i];
        }
    }

    return {BestIndex};
}
