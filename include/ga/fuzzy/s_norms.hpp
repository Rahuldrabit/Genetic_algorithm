#pragma once

#include <algorithm>
#include <cmath>
#include "ga/fuzzy/fuzzy_types.hpp"

namespace ga {
namespace fuzzy {

inline double applySNorm(SNormType type, double a, double b, double gamma = 0.5) {
    a = std::clamp(a, 0.0, 1.0);
    b = std::clamp(b, 0.0, 1.0);

    switch (type) {
        case SNormType::Maximum:
            return std::max(a, b);

        case SNormType::AlgebraicSum:
            return a + b - a * b;

        case SNormType::Lukasiewicz:
            return std::min(1.0, a + b);

        case SNormType::DrasticSum:
            if (std::abs(a) < 1e-12) return b;
            if (std::abs(b) < 1e-12) return a;
            return 1.0;

        case SNormType::HamacherSum: {
            const double denom = 1.0 - (1.0 - gamma) * a * b;
            return denom > 1e-12 ? (a + b - (2.0 - gamma) * a * b) / denom : 1.0;
        }

        case SNormType::EinsteinSum: {
            const double denom = 1.0 + a * b;
            return denom > 1e-12 ? (a + b) / denom : 1.0;
        }
    }
    return std::max(a, b);
}

} // namespace fuzzy
} // namespace ga
