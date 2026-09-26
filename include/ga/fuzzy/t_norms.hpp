#pragma once

#include <algorithm>
#include <cmath>
#include "ga/fuzzy/fuzzy_types.hpp"

namespace ga {
namespace fuzzy {

inline double applyTNorm(TNormType type, double a, double b, double gamma = 0.5) {
    a = std::clamp(a, 0.0, 1.0);
    b = std::clamp(b, 0.0, 1.0);

    switch (type) {
        case TNormType::Minimum:
            return std::min(a, b);

        case TNormType::AlgebraicProduct:
            return a * b;

        case TNormType::Lukasiewicz:
            return std::max(0.0, a + b - 1.0);

        case TNormType::DrasticProduct:
            if (std::abs(a - 1.0) < 1e-12) return b;
            if (std::abs(b - 1.0) < 1e-12) return a;
            return 0.0;

        case TNormType::HamacherProduct: {
            const double denom = gamma + (1.0 - gamma) * (a + b - a * b);
            return denom > 1e-12 ? (a * b) / denom : 0.0;
        }

        case TNormType::EinsteinProduct: {
            const double denom = 2.0 - (a + b - a * b);
            return denom > 1e-12 ? (a * b) / denom : 0.0;
        }
    }
    return std::min(a, b);
}

} // namespace fuzzy
} // namespace ga
