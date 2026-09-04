#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "ga/metaheuristics/common.hpp"

namespace ga {
namespace aco {

class DenseGraph {
public:
    explicit DenseGraph(std::vector<std::vector<double>> costs,
                        bool symmetric = true);

    std::size_t size() const noexcept { return size_; }
    double cost(std::size_t from, std::size_t to) const noexcept {
        return costs_[from * size_ + to];
    }
    bool symmetric() const noexcept { return symmetric_; }

private:
    std::size_t size_ = 0;
    std::vector<double> costs_;
    bool symmetric_ = true;
};

enum class AntColonyVariant {
    AntSystem,
    ElitistAntSystem,
    RankBasedAntSystem,
    AntColonySystem,
    MaxMinAntSystem
};

struct AntColonyConfig {
    std::size_t ants = 30;
    std::size_t iterations = 100;
    double alpha = 1.0;
    double beta = 2.0;
    double evaporation = 0.1;
    double depositScale = 1.0;
    double initialPheromone = 1.0;
    double elitistWeight = 2.0;
    std::size_t rankCount = 6;
    double exploitationProbability = 0.9;
    double localEvaporation = 0.1;
    std::size_t candidateListSize = 20;
    double minPheromone = 0.0;
    double maxPheromone = 0.0;
    AntColonyVariant variant = AntColonyVariant::AntSystem;
    unsigned seed = 0;
    std::shared_ptr<const ga::metaheuristics::IAdaptiveController> controller;
};

struct AntColonyResult {
    std::vector<std::size_t> bestTour;
    double bestCost = 0.0;
    std::vector<double> bestCostHistory;
    std::size_t evaluations = 0;
    std::size_t iterations = 0;
};

class AntColonyOptimizer {
public:
    explicit AntColonyOptimizer(AntColonyConfig config = {});

    AntColonyResult solve(const DenseGraph& graph) const;
    const AntColonyConfig& config() const noexcept { return config_; }

private:
    AntColonyConfig config_;
};

} // namespace aco
} // namespace ga
