#pragma once
/// Visualization Tools
///
/// Exports data from a GA run to CSV files suitable for plotting in Python,
/// Excel, or any CSV-aware tool.
///
/// FitnessCurveExporter   – best / avg fitness per generation.
/// ParetoFrontExporter    – 2-D and N-D Pareto front points.
/// DiversityPlotExporter  – population diversity per generation.
///
/// Python quick-start (after export):
///   import pandas as pd, matplotlib.pyplot as plt
///   df = pd.read_csv("fitness_curve.csv")
///   plt.plot(df["generation"], df["best"], label="best")
///   plt.plot(df["generation"], df["avg"],  label="avg")
///   plt.legend(); plt.show()

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <iomanip>

namespace ga {
namespace viz {

// ============================================================================
// FitnessCurveExporter
// ============================================================================

class FitnessCurveExporter {
public:
    explicit FitnessCurveExporter(std::string filepath = "fitness_curve.csv")
        : filepath_(std::move(filepath)) {}

    /// @p bestHistory  best fitness per generation
    /// @p avgHistory   average fitness per generation
    void save(const std::vector<double>& bestHistory,
              const std::vector<double>& avgHistory) const {
        std::ofstream out(filepath_);
        if (!out) throw std::runtime_error("FitnessCurveExporter: cannot open '" + filepath_ + "'");
        out << "generation,best,avg\n";
        for (size_t i = 0; i < bestHistory.size(); ++i) {
            double avg = i < avgHistory.size() ? avgHistory[i] : 0.0;
            out << i << ',' << std::setprecision(10) << bestHistory[i] << ',' << avg << '\n';
        }
    }

    const std::string& filepath() const { return filepath_; }

private:
    std::string filepath_;
};

// ============================================================================
// ParetoFrontExporter
// ============================================================================

class ParetoFrontExporter {
public:
    explicit ParetoFrontExporter(std::string filepath = "pareto_front.csv")
        : filepath_(std::move(filepath)) {}

    /// @p front  Vector of objective vectors (one per Pareto-optimal solution).
    void save(const std::vector<std::vector<double>>& front) const {
        if (front.empty()) return;
        std::ofstream out(filepath_);
        if (!out) throw std::runtime_error("ParetoFrontExporter: cannot open '" + filepath_ + "'");

        // Header
        for (size_t j = 0; j < front[0].size(); ++j) {
            if (j) out << ',';
            out << "obj" << j;
        }
        out << '\n';

        // Data
        for (const auto& obj : front) {
            for (size_t j = 0; j < obj.size(); ++j) {
                if (j) out << ',';
                out << std::setprecision(10) << obj[j];
            }
            out << '\n';
        }
    }

    const std::string& filepath() const { return filepath_; }

private:
    std::string filepath_;
};

// ============================================================================
// DiversityPlotExporter
// ============================================================================

class DiversityPlotExporter {
public:
    explicit DiversityPlotExporter(std::string filepath = "diversity.csv")
        : filepath_(std::move(filepath)) {}

    void save(const std::vector<double>& diversityHistory) const {
        std::ofstream out(filepath_);
        if (!out) throw std::runtime_error("DiversityPlotExporter: cannot open '" + filepath_ + "'");
        out << "generation,diversity\n";
        for (size_t i = 0; i < diversityHistory.size(); ++i)
            out << i << ',' << std::setprecision(10) << diversityHistory[i] << '\n';
    }

    const std::string& filepath() const { return filepath_; }

private:
    std::string filepath_;
};

// ============================================================================
// Convenience: export everything in one call
// ============================================================================

struct ExportBundle {
    std::vector<double>              bestHistory;
    std::vector<double>              avgHistory;
    std::vector<double>              diversityHistory;
    std::vector<std::vector<double>> paretoFront;   ///< empty if not multi-obj
};

inline void exportAll(const ExportBundle& bundle,
                      const std::string& prefix = "") {
    FitnessCurveExporter(prefix + "fitness_curve.csv")
        .save(bundle.bestHistory, bundle.avgHistory);

    if (!bundle.diversityHistory.empty())
        DiversityPlotExporter(prefix + "diversity.csv")
            .save(bundle.diversityHistory);

    if (!bundle.paretoFront.empty())
        ParetoFrontExporter(prefix + "pareto_front.csv")
            .save(bundle.paretoFront);
}

} // namespace viz
} // namespace ga
