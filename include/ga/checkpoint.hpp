#pragma once
/// Checkpointing – save and restore a GA population and run state.
///
/// File format: plain text (human-readable, easy to parse):
///   Line 1:  "GACHECKPOINT v1"
///   Line 2:  generation <N>
///   Line 3:  seed <S>
///   Line 4+: individual <fitness> <gene0> <gene1> … <geneN-1>
///   Last:    "END"

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iomanip>

namespace ga {
namespace checkpoint {

// ============================================================================
// CheckpointData – the serialisable state of a GA run
// ============================================================================

struct IndividualSnapshot {
    std::vector<double> genes;
    double fitness = 0.0;
};

struct CheckpointData {
    int generation = 0;
    unsigned seed  = 0;
    std::vector<IndividualSnapshot> population;
};

// ============================================================================
// save / load
// ============================================================================

/// Save @p data to @p filepath (overwrites existing file).
inline void save(const std::string& filepath, const CheckpointData& data) {
    std::ofstream out(filepath);
    if (!out)
        throw std::runtime_error("Checkpoint: cannot open '" + filepath + "' for writing");

    out << "GACHECKPOINT v1\n";
    out << "generation " << data.generation << "\n";
    out << "seed "       << data.seed       << "\n";

    out << std::setprecision(17);
    for (const auto& ind : data.population) {
        out << "individual " << ind.fitness;
        for (double g : ind.genes) out << ' ' << g;
        out << "\n";
    }
    out << "END\n";
}

/// Load a checkpoint from @p filepath.
inline CheckpointData load(const std::string& filepath) {
    std::ifstream in(filepath);
    if (!in)
        throw std::runtime_error("Checkpoint: cannot open '" + filepath + "' for reading");

    std::string line;
    std::getline(in, line);
    if (line != "GACHECKPOINT v1")
        throw std::runtime_error("Checkpoint: unrecognised format");

    CheckpointData data;
    while (std::getline(in, line)) {
        if (line.empty() || line == "END") break;
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;

        if (tag == "generation") {
            ss >> data.generation;
        } else if (tag == "seed") {
            ss >> data.seed;
        } else if (tag == "individual") {
            IndividualSnapshot ind;
            ss >> ind.fitness;
            double g;
            while (ss >> g) ind.genes.push_back(g);
            data.population.push_back(std::move(ind));
        }
    }
    return data;
}

// ============================================================================
// CheckpointManager – wraps periodic auto-saving
// ============================================================================

class CheckpointManager {
public:
    /// @p filepath        File to write checkpoints to.
    /// @p saveEveryNGens  Save every N generations (0 = disabled).
    explicit CheckpointManager(std::string filepath, int saveEveryNGens = 10)
        : filepath_(std::move(filepath)), interval_(saveEveryNGens) {}

    /// Called each generation. Saves automatically if the interval is due.
    void tick(int generation, const CheckpointData& data) {
        if (interval_ > 0 && generation % interval_ == 0)
            save(filepath_, data);
    }

    void forceSave(const CheckpointData& data) {
        save(filepath_, data);
    }

    CheckpointData tryLoad() const {
        return load(filepath_);
    }

    bool fileExists() const {
        std::ifstream f(filepath_);
        return f.good();
    }

private:
    std::string filepath_;
    int         interval_;
};

} // namespace checkpoint
} // namespace ga
