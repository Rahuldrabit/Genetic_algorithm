#pragma once

#include <cstddef>
#include "ga/core/evaluation.hpp"
#include "ga/core/genome.hpp"

namespace ga {

class IProblem {
public:
    virtual ~IProblem() = default;
    virtual std::size_t objectiveCount() const = 0;
    virtual Evaluation evaluate(const IGenome& genome) const = 0;
};

} // namespace ga
