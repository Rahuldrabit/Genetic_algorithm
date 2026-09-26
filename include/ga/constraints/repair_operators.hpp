#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace ga {
namespace constraints {

// Repairs box bounds by clamping or reflecting
inline void repairBoxBounds(std::vector<double>& genes, double lower, double upper) {
    for (double& g : genes) {
        g = std::clamp(g, lower, upper);
    }
}

// Repairs a chromosome so that the sum of non-negative components equals targetSum (Simplex constraint)
inline void repairSimplex(std::vector<double>& genes, double targetSum = 1.0) {
    if (genes.empty()) return;
    for (double& g : genes) {
        if (g < 0.0) g = 0.0;
    }
    double currentSum = std::accumulate(genes.begin(), genes.end(), 0.0);
    if (currentSum > 1e-12) {
        double scale = targetSum / currentSum;
        for (double& g : genes) {
            g *= scale;
        }
    } else {
        double equalShare = targetSum / static_cast<double>(genes.size());
        for (double& g : genes) {
            g = equalShare;
        }
    }
}

// Repairs linear inequality: a^T * x <= b. If violated, projects along normal vector 'a'
inline void repairLinearInequality(
    std::vector<double>& genes,
    const std::vector<double>& a,
    double b) {
    if (genes.size() != a.size()) return;

    double dot = 0.0;
    double aNormSq = 0.0;
    for (std::size_t i = 0; i < genes.size(); ++i) {
        dot += a[i] * genes[i];
        aNormSq += a[i] * a[i];
    }

    if (dot > b && aNormSq > 1e-12) {
        double violation = dot - b;
        double lambda = violation / aNormSq;
        for (std::size_t i = 0; i < genes.size(); ++i) {
            genes[i] -= lambda * a[i];
        }
    }
}

// Repairs permutation chromosome (0..N-1) by replacing duplicate elements with missing ones
inline void repairPermutation(std::vector<int>& perm) {
    const std::size_t n = perm.size();
    if (n == 0) return;

    std::vector<bool> seen(n, false);
    std::vector<std::size_t> duplicateIndices;

    for (std::size_t i = 0; i < n; ++i) {
        int val = perm[i];
        if (val >= 0 && static_cast<std::size_t>(val) < n && !seen[static_cast<std::size_t>(val)]) {
            seen[static_cast<std::size_t>(val)] = true;
        } else {
            duplicateIndices.push_back(i);
        }
    }

    std::vector<int> missing;
    for (std::size_t v = 0; v < n; ++v) {
        if (!seen[v]) {
            missing.push_back(static_cast<int>(v));
        }
    }

    for (std::size_t k = 0; k < duplicateIndices.size() && k < missing.size(); ++k) {
        perm[duplicateIndices[k]] = missing[k];
    }
}

} // namespace constraints
} // namespace ga
