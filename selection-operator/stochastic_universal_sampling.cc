#include "stochastic_universal_sampling.h"
#include <algorithm>
#include <random>
#include <iostream>

StochasticUniversalSampling::StochasticUniversalSampling() 
    : SelectionOperator("StochasticUniversalSampling") {}

std::vector<Individual> StochasticUniversalSampling::select(const std::vector<Individual>& population, size_t count) {
    std::vector<Individual> selected;
    selected.reserve(count);
    
    if (population.empty() || count == 0) {
        return selected;
    }
    
    operation_count++;
    
    // Calculate the total fitness
    double total_fitness = 0.0;
    for (const auto& individual : population) {
        total_fitness += individual.fitness;
    }
    
    if (total_fitness <= 0.0) {
        // If no positive fitness, select randomly
        std::uniform_int_distribution<> dis(0, population.size() - 1);
        for (size_t i = 0; i < count; ++i) {
            selected.push_back(population[dis(rng)]);
        }
        return selected;
    }
    
    // Calculate the distance between selection points
    double distance = total_fitness / count;
    
    // Random starting point
    std::uniform_real_distribution<> start_dis(0.0, distance);
    double start_point = start_dis(rng);
    
    // Select individuals using evenly spaced pointers
    double cumulative_fitness = 0.0;
    size_t pop_index = 0;
    
    for (size_t i = 0; i < count; ++i) {
        double selection_point = start_point + i * distance;
        
        // Find the individual at this selection point
        while (pop_index < population.size() && cumulative_fitness < selection_point) {
            cumulative_fitness += population[pop_index].fitness;
            if (cumulative_fitness < selection_point) {
                pop_index++;
            }
        }
        
        if (pop_index < population.size()) {
            selected.push_back(population[pop_index]);
        } else if (!population.empty()) {
            selected.push_back(population.back());
        }
    }
    
    return selected;
}

std::vector<unsigned int> StochasticUniversalSampling::selectIndices(std::vector<Individual>& Population, 
                                                                    unsigned int NumSelections) {
    StochasticUniversalSampling selector;
    std::vector<Individual> selected = selector.select(Population, NumSelections);
    
    // Convert back to indices
    std::vector<unsigned int> indices;
    indices.reserve(NumSelections);
    
    for (const auto& sel : selected) {
        for (size_t i = 0; i < Population.size(); ++i) {
            if (&Population[i] == &sel) {
                indices.push_back(i);
                break;
            }
        }
    }
    
    return indices;
}

// Legacy function for backward compatibility
std::vector<unsigned int> StochasticUniversalSamplingLegacy(std::vector<Individual>& Population, 
                                                           unsigned int NumSelections) {
    std::vector<unsigned int> SelectedIndices;
    unsigned int PopulationSize = Population.size();

    if (PopulationSize == 0 || NumSelections == 0) {
        std::cerr << "Error: Population is empty or number of selections is zero." << std::endl;
        return {};
    }

    // Calculate the total fitness
    double TotalFitness = 0.0;
    for (const auto& individual : Population) {
        TotalFitness += individual.fitness;
    }

    if (TotalFitness <= 0.0) {
        std::cerr << "Error: Total fitness is non-positive." << std::endl;
        return {};
    }

    // Calculate the distance between selection points
    double Distance = TotalFitness / NumSelections;
    double StartPoint = static_cast<double>(rand()) / RAND_MAX * Distance;

    // Select individuals using evenly spaced pointers
    double CumulativeFitness = 0.0;
    unsigned int PopIndex = 0;

    for (unsigned int i = 0; i < NumSelections; ++i) {
        double SelectionPoint = StartPoint + i * Distance;

        while (PopIndex < PopulationSize && CumulativeFitness < SelectionPoint) {
            CumulativeFitness += Population[PopIndex].fitness;
            PopIndex++;
        }

        if (PopIndex > 0) {
            SelectedIndices.push_back(PopIndex - 1);
        } else {
            SelectedIndices.push_back(0);
        }
    }

    return SelectedIndices;
}
