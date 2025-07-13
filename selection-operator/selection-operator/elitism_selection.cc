#include "elitism_selection.h"
#include <algorithm>
#include <iostream>

ElitismSelection::ElitismSelection() 
    : SelectionOperator("Elitism") {}

std::vector<Individual> ElitismSelection::select(const std::vector<Individual>& population, size_t count) {
    std::vector<Individual> selected;
    
    if (population.empty() || count == 0) {
        return selected;
    }
    
    operation_count++;
    
    // Create a copy and sort by fitness in descending order (best first)
    std::vector<Individual> sorted_population = population;
    std::sort(sorted_population.begin(), sorted_population.end(),
        [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
        });
    
    // Select the top 'count' individuals
    size_t num_to_select = std::min(count, sorted_population.size());
    selected.reserve(num_to_select);
    
    for (size_t i = 0; i < num_to_select; ++i) {
        selected.push_back(sorted_population[i]);
    }
    
    return selected;
}

std::vector<unsigned int> ElitismSelection::selectIndices(std::vector<Individual>& Population, 
                                                         unsigned int NumElites) {
    std::vector<unsigned int> SelectedIndices;
    unsigned int PopulationSize = Population.size();

    if (PopulationSize == 0 || NumElites == 0) {
        return SelectedIndices;
    }

    // Sort individuals based on fitness in descending order
    std::vector<unsigned int> Indices(PopulationSize);
    for (unsigned int i = 0; i < PopulationSize; ++i) {
        Indices[i] = i;
    }

    std::sort(Indices.begin(), Indices.end(), [&](unsigned int a, unsigned int b) {
        return Population[a].fitness > Population[b].fitness;
    });

    // Select the top NumElites individuals
    unsigned int numToSelect = std::min(NumElites, PopulationSize);
    SelectedIndices.reserve(numToSelect);
    
    for (unsigned int i = 0; i < numToSelect; ++i) {
        SelectedIndices.push_back(Indices[i]);
    }

    return SelectedIndices;
}

// Legacy function for backward compatibility
std::vector<unsigned int> ElitismSelectionLegacy(std::vector<Individual>& Population, 
                                                unsigned int NumElites) {
    std::vector<unsigned int> SelectedIndices;
    unsigned int PopulationSize = Population.size();

    if (PopulationSize == 0 || NumElites == 0) {
        std::cerr << "Error: Population is empty or number of elites is zero." << std::endl;
        return {};
    }

    // Sort individuals based on fitness in descending order
    std::vector<unsigned int> Indices(PopulationSize);
    for (unsigned int i = 0; i < PopulationSize; ++i) {
        Indices[i] = i;
    }

    std::sort(Indices.begin(), Indices.end(), [&](unsigned int a, unsigned int b) {
        return Population[a].fitness > Population[b].fitness;
    });

    // Select the top NumElites individuals
    for (unsigned int i = 0; i < NumElites && i < PopulationSize; ++i) {
        SelectedIndices.push_back(Indices[i]);
    }

    return SelectedIndices;
}
