#pragma once

#include <algorithm>
#include <cmath>
#include "ga/fuzzy/fuzzy_types.hpp"

namespace ga {
namespace fuzzy {

inline double applyComplement(ComplementType type, double a, double parameter = 1.0) {
    a = std::clamp(a, 0.0, 1.0);

    switch (type) {
        case ComplementType::Standard:
            return 1.0 - a;

        case ComplementType::Sugeno: {
            // parameter is lambda > -1
            const double lambda = parameter > -1.0 ? parameter : 0.0;
            return (1.0 - a) / (1.0 + lambda * a);
        }

        case ComplementType::Yager: {
            // parameter is w > 0
            const double w = parameter > 0.0 ? parameter : 1.0;
            return std::pow(1.0 - std::pow(a, w), 1.0 / w);
        }
    }
    return 1.0 - a;
}

} // namespace fuzzy
} // namespace ga
