#pragma once
/// Experiment Tracking
///
/// ExperimentLogger records parameters, per-generation statistics, and the
/// best solution.  Results can be flushed to CSV or a human-readable text file.
///
/// Usage:
///   ga::experiment::ExperimentLogger log("my_run");
///   log.setParam("populationSize", 50);
///   log.setParam("mutation", "gaussian");
///   // inside the generation loop:
///   log.record(gen, bestFitness, avgFitness);
///   // at the end:
///   log.setBestGenes(bestGenes);
///   log.saveText("results.txt");
///   log.saveCSV("results.csv");

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <chrono>
#include <ctime>

namespace ga {
namespace experiment {

// ============================================================================
// GenerationRecord
// ============================================================================

struct GenerationRecord {
    int    generation  = 0;
    double bestFitness = 0.0;
    double avgFitness  = 0.0;
    double diversity   = 0.0;   ///< optional (set to 0 if not tracked)
};

// ============================================================================
// ExperimentLogger
// ============================================================================

class ExperimentLogger {
public:
    explicit ExperimentLogger(std::string experimentName = "ga_run")
        : name_(std::move(experimentName)) {
        auto now = std::chrono::system_clock::now();
        auto t   = std::chrono::system_clock::to_time_t(now);
        std::ostringstream ts;
        ts << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
        timestamp_ = ts.str();
    }

    // ---- parameter recording ----
    void setParam(const std::string& key, double value) {
        params_[key] = std::to_string(value);
    }
    void setParam(const std::string& key, int value) {
        params_[key] = std::to_string(value);
    }
    void setParam(const std::string& key, const std::string& value) {
        params_[key] = value;
    }

    // ---- per-generation recording ----
    void record(int generation, double bestFitness, double avgFitness,
                double diversity = 0.0) {
        history_.push_back({generation, bestFitness, avgFitness, diversity});
    }

    // ---- final result ----
    void setBestGenes(const std::vector<double>& genes) { bestGenes_ = genes; }
    void setBestFitness(double f) { bestFitness_ = f; }

    // ---- export ----
    void saveText(const std::string& filepath) const {
        std::ofstream out(filepath);
        if (!out) throw std::runtime_error("ExperimentLogger: cannot open '" + filepath + "'");

        out << "=== Experiment: " << name_ << " ===\n";
        out << "Timestamp: " << timestamp_ << "\n\n";

        out << "--- Parameters ---\n";
        for (const auto& kv : params_)
            out << kv.first << " = " << kv.second << "\n";

        out << "\n--- Best Result ---\n";
        out << "Best fitness: " << bestFitness_ << "\n";
        if (!bestGenes_.empty()) {
            out << "Best genes:  ";
            for (double g : bestGenes_) out << g << ' ';
            out << "\n";
        }

        out << "\n--- Generation History ---\n";
        out << "gen\tbest\tavg\tdiversity\n";
        for (const auto& r : history_)
            out << r.generation << '\t' << r.bestFitness << '\t'
                << r.avgFitness << '\t' << r.diversity << '\n';
    }

    void saveCSV(const std::string& filepath) const {
        std::ofstream out(filepath);
        if (!out) throw std::runtime_error("ExperimentLogger: cannot open '" + filepath + "'");

        out << "generation,best_fitness,avg_fitness,diversity\n";
        for (const auto& r : history_)
            out << r.generation << ',' << r.bestFitness << ','
                << r.avgFitness << ',' << r.diversity << '\n';
    }

    // ---- accessors ----
    const std::vector<GenerationRecord>& history() const { return history_; }
    double bestFitness() const { return bestFitness_; }
    const std::vector<double>& bestGenes() const { return bestGenes_; }
    const std::string& name() const { return name_; }

private:
    std::string name_;
    std::string timestamp_;
    std::map<std::string, std::string> params_;
    std::vector<GenerationRecord> history_;
    std::vector<double> bestGenes_;
    double bestFitness_ = -1e300;
};

} // namespace experiment
} // namespace ga
