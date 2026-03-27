#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ga/config.hpp"

namespace ga {
namespace tracking {

class ExperimentTracker {
public:
    explicit ExperimentTracker(std::string runId)
        : runId_(std::move(runId)) {}

    void writeConfig(const ga::Config& cfg, const std::string& path) const {
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("Unable to write experiment config: " + path);
        }
        out << "run_id=" << runId_ << "\n";
        out << "population_size=" << cfg.populationSize << "\n";
        out << "generations=" << cfg.generations << "\n";
        out << "dimension=" << cfg.dimension << "\n";
        out << "crossover_rate=" << cfg.crossoverRate << "\n";
        out << "mutation_rate=" << cfg.mutationRate << "\n";
        out << "lower_bound=" << cfg.bounds.lower << "\n";
        out << "upper_bound=" << cfg.bounds.upper << "\n";
        out << "elite_ratio=" << cfg.eliteRatio << "\n";
        out << "seed=" << cfg.seed << "\n";
    }

    void writeHistoryCSV(const ga::Result& result, const std::string& path) const {
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("Unable to write history CSV: " + path);
        }
        out << "generation,best,average\n";
        const std::size_t n = std::min(result.bestHistory.size(), result.avgHistory.size());
        for (std::size_t i = 0; i < n; ++i) {
            out << i << "," << result.bestHistory[i] << "," << result.avgHistory[i] << "\n";
        }
    }

    void writeBestSolutionCSV(const ga::Result& result, const std::string& path) const {
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("Unable to write best solution CSV: " + path);
        }
        out << "index,value\n";
        for (std::size_t i = 0; i < result.bestGenes.size(); ++i) {
            out << i << "," << result.bestGenes[i] << "\n";
        }
    }

private:
    std::string runId_;
};

} // namespace tracking
} // namespace ga
