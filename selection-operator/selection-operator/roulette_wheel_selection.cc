#include "roulette_wheel_selection.h"
#include <random>
#include <algorithm>

std::vector<Individual> RouletteWheelSelection::select(const std::vector<Individual>& population, size_t count) {
    std::vector<Individual> selected;
    selected.reserve(count);
    
    if (population.empty()) {
        return selected;
    }
    
    double total_fitness = 0.0;
    for (const auto& individual : population) {
        total_fitness += individual.fitness;
    }
    
    if (total_fitness <= 0.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, population.size() - 1);
        
        for (size_t i = 0; i < count; ++i) {
            selected.push_back(population[dis(gen)]);
        }
        return selected;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, total_fitness);
    
    for (size_t i = 0; i < count; ++i) {
        double random_value = dis(gen);
        double cumulative_fitness = 0.0;
        
        for (const auto& individual : population) {
            cumulative_fitness += individual.fitness;
            if (cumulative_fitness >= random_value) {
                selected.push_back(individual);
                break;
            }
        }
    }
    
    return selected;
}

// Legacy function for backward compatibility
std::vector<unsigned int> RouletteWheelSelectionLegacy(std::vector<Individual>& Population, unsigned int NumSelections) {
    std::vector<unsigned int> SelectedIndices;
    unsigned int PopulationSize = Population.size();

    if (PopulationSize == 0) {
        std::cerr << "Error: Population is empty." << std::endl;
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

    // Select individuals based on their fitness proportion
    for (unsigned int i = 0; i < NumSelections; ++i) {
        double RandomValue = static_cast<double>(rand()) / RAND_MAX * TotalFitness;
        double CumulativeFitness = 0.0;

        for (unsigned int j = 0; j < PopulationSize; ++j) {
            CumulativeFitness += Population[j].fitness;
            if (CumulativeFitness >= RandomValue) {
                SelectedIndices.push_back(j);
                break;
            }
        }
    }

    return SelectedIndices;
}
