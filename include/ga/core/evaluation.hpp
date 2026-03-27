#pragma once

#include <vector>

namespace ga {

// Generic evaluation container used across single- and multi-objective algorithms.
struct Evaluation {
    std::vector<double> objectives;
    bool feasible = true;
    double penalty = 0.0;
};

} // namespace ga
