#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ga {
namespace visualization {

inline void exportFitnessCurveCSV(const std::vector<double>& best,
                                  const std::vector<double>& avg,
                                  const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open fitness CSV: " + path);
    }
    out << "generation,best,avg\n";
    const std::size_t n = std::min(best.size(), avg.size());
    for (std::size_t i = 0; i < n; ++i) {
        out << i << "," << best[i] << "," << avg[i] << "\n";
    }
}

inline void exportParetoFrontCSV(const std::vector<std::vector<double>>& objectives,
                                 const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open pareto CSV: " + path);
    }
    if (objectives.empty()) {
        out << "objective0\n";
        return;
    }
    const std::size_t dims = objectives.front().size();
    for (std::size_t d = 0; d < dims; ++d) {
        out << "objective" << d;
        if (d + 1 < dims) {
            out << ",";
        }
    }
    out << "\n";

    for (const auto& vec : objectives) {
        for (std::size_t d = 0; d < vec.size(); ++d) {
            out << vec[d];
            if (d + 1 < vec.size()) {
                out << ",";
            }
        }
        out << "\n";
    }
}

inline void exportDiversityCSV(const std::vector<double>& diversity, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open diversity CSV: " + path);
    }
    out << "generation,diversity\n";
    for (std::size_t i = 0; i < diversity.size(); ++i) {
        out << i << "," << diversity[i] << "\n";
    }
}

} // namespace visualization
} // namespace ga
