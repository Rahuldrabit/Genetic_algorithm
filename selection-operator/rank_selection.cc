#include "rank_selection.h"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iostream>

RankSelection::RankSelection(double pressure) 
    : SelectionOperator("Rank"), selection_pressure(pressure) {
    if (pressure < 1.0) {
        throw std::invalid_argument("Selection pressure must be >= 1.0");
    }
}

void RankSelection::setSelectionPressure(double pressure) {
    if (pressure < 1.0) {
        throw std::invalid_argument("Selection pressure must be >= 1.0");
    }
    selection_pressure = pressure;
}

std::vector<Individual> RankSelection::select(const std::vector<Individual>& population, size_t count) {
    std::vector<Individual> selected;
    selected.reserve(count);
    
    if (population.empty()) {
        return selected;
    }
    
    operation_count++;
    
    // Create a copy and sort by fitness (ascending order, worst to best)
    std::vector<Individual> sorted_population = population;
    std::sort(sorted_population.begin(), sorted_population.end(),
        [](const Individual& a, const Individual& b) {
            return a.fitness < b.fitness;
        });
    
    // Calculate selection probabilities based on rank
    std::vector<double> probabilities;
    probabilities.reserve(sorted_population.size());
    
    size_t n = sorted_population.size();
    for (size_t i = 0; i < n; ++i) {
        // Rank i goes from 0 (worst) to n-1 (best)
        // Probability formula: P(i) = (2-SP + 2*(SP-1)*i/(n-1)) / n
        // where SP is selection pressure
        double prob;
        if (n == 1) {
            prob = 1.0;
        } else {
            prob = (2.0 - selection_pressure + 2.0 * (selection_pressure - 1.0) * i / (n - 1)) / n;
        }
        probabilities.push_back(prob);
    }
    
    // Calculate cumulative probabilities
    std::vector<double> cumulative_probs;
    cumulative_probs.reserve(n);
    double sum = 0.0;
    for (double prob : probabilities) {
        sum += prob;
        cumulative_probs.push_back(sum);
    }
    
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (size_t i = 0; i < count; ++i) {
        double random_value = dis(rng);
        
        auto it = std::lower_bound(cumulative_probs.begin(), cumulative_probs.end(), random_value);
        size_t index = std::distance(cumulative_probs.begin(), it);
        
        if (index < sorted_population.size()) {
            selected.push_back(sorted_population[index]);
        } else {
            selected.push_back(sorted_population.back());
        }
    }
    
    return selected;
}

std::vector<unsigned int> RankSelection::selectIndices(std::vector<Individual>& Population, 
                                                      unsigned int NumSelections) {
    RankSelection selector;
    std::vector<Individual> selected = selector.select(Population, NumSelections);
    
    // Convert back to indices (this is a simplified approach)
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
std::vector<unsigned int> RankSelectionLegacy(std::vector<Individual>& Population, unsigned int NumSelections) {
    std::vector<unsigned int> SelectedIndices;
    unsigned int PopulationSize = Population.size();

    if (PopulationSize == 0) {
        std::cerr << "Error: Population is empty." << std::endl;
        return {};
    }

    // Create a vector of indices and sort them based on fitness
    std::vector<unsigned int> Indices(PopulationSize);
    for (unsigned int i = 0; i < PopulationSize; ++i) {
        Indices[i] = i;
    }

    std::sort(Indices.begin(), Indices.end(), [&](unsigned int a, unsigned int b) {
        return Population[a].fitness > Population[b].fitness;
    });

    // Calculate total rank sum for selection probability
    double TotalRankSum = (PopulationSize * (PopulationSize + 1)) / 2.0;

    // Select individuals based on their rank
    for (unsigned int i = 0; i < NumSelections; ++i) {
        double RandomValue = static_cast<double>(rand()) / RAND_MAX * TotalRankSum;
        double CumulativeRank = 0.0;

        for (unsigned int j = 0; j < PopulationSize; ++j) {
            // Higher fitness gets higher rank (PopulationSize for best, 1 for worst)
            CumulativeRank += (PopulationSize - j);
            if (CumulativeRank >= RandomValue) {
                SelectedIndices.push_back(Indices[j]);
                break;
            }
        }
    }

    return SelectedIndices;
}
