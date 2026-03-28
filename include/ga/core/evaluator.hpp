#pragma once

#include <vector>
#include "ga/core/genome.hpp"
#include "ga/core/evaluation.hpp"

namespace ga {
namespace core {

// Abstract evaluator interface: maps a genome to an Evaluation result.
// Implementations can evaluate in-process, via threads, or via remote workers.
class IEvaluator {
public:
    virtual ~IEvaluator() = default;

    // Evaluate a single genome and return its fitness/objectives.
    virtual Evaluation evaluate(const IGenome& genome) const = 0;

    // Batch evaluate a population of genomes; default calls evaluate() for each.
    virtual std::vector<Evaluation> evaluateBatch(const std::vector<const IGenome*>& genomes) const {
        std::vector<Evaluation> results;
        results.reserve(genomes.size());
        for (const IGenome* g : genomes) {
            if (g) {
                results.push_back(evaluate(*g));
            } else {
                results.push_back({});
            }
        }
        return results;
    }
};

} // namespace core
} // namespace ga
