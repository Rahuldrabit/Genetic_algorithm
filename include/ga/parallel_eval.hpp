#pragma once
/// Parallel Evaluation – header-only thread-pool for fitness evaluation.
///
/// Wraps std::thread / std::async to evaluate a population's fitness
/// concurrently.  Drop-in replacement for sequential loops.
///
/// Usage:
///   ga::parallel::ParallelEval eval(/*threads=*/4);
///   eval.evaluate(population, fitnessFunc);

#include <vector>
#include <functional>
#include <thread>
#include <future>
#include <algorithm>
#include <cmath>

namespace ga {
namespace parallel {

// ============================================================================
// ParallelEval – evaluate a vector of individuals concurrently
// ============================================================================

class ParallelEval {
public:
    /// @p numThreads  0 = use hardware_concurrency
    explicit ParallelEval(unsigned numThreads = 0)
        : numThreads_(numThreads == 0
                      ? std::max(1u, std::thread::hardware_concurrency())
                      : numThreads) {}

    unsigned numThreads() const { return numThreads_; }

    /// Evaluate fitness for every individual in @p genes.
    /// @p genes       Vector of gene vectors (one per individual).
    /// @p fitnessFunc Fitness function: genes → scalar (higher = better).
    /// Returns a vector of fitness values aligned with @p genes.
    std::vector<double> evaluate(
        const std::vector<std::vector<double>>& genes,
        const std::function<double(const std::vector<double>&)>& fitnessFunc) const
    {
        const size_t n = genes.size();
        std::vector<double> results(n, 0.0);

        if (n == 0) return results;

        // Partition work into numThreads_ chunks
        const size_t chunkSize = (n + numThreads_ - 1) / numThreads_;
        std::vector<std::future<void>> futures;
        futures.reserve(numThreads_);

        for (size_t t = 0; t < numThreads_; ++t) {
            size_t begin = t * chunkSize;
            size_t end   = std::min(begin + chunkSize, n);
            if (begin >= end) break;

            futures.push_back(std::async(std::launch::async,
                [&genes, &fitnessFunc, &results, begin, end]() {
                    for (size_t i = begin; i < end; ++i)
                        results[i] = fitnessFunc(genes[i]);
                }));
        }

        for (auto& f : futures) f.get();
        return results;
    }

    /// Convenience overload: evaluate a population stored as a flat struct
    /// with a `.genes` member and a `.fitness` field to be written.
    template <typename Individual>
    void evaluatePopulation(
        std::vector<Individual>& pop,
        const std::function<double(const std::vector<double>&)>& fitnessFunc) const
    {
        const size_t n = pop.size();
        if (n == 0) return;

        const size_t chunkSize = (n + numThreads_ - 1) / numThreads_;
        std::vector<std::future<void>> futures;
        futures.reserve(numThreads_);

        for (size_t t = 0; t < numThreads_; ++t) {
            size_t begin = t * chunkSize;
            size_t end   = std::min(begin + chunkSize, n);
            if (begin >= end) break;

            futures.push_back(std::async(std::launch::async,
                [&pop, &fitnessFunc, begin, end]() {
                    for (size_t i = begin; i < end; ++i)
                        pop[i].fitness = fitnessFunc(pop[i].genes);
                }));
        }

        for (auto& f : futures) f.get();
    }

private:
    unsigned numThreads_;
};

} // namespace parallel
} // namespace ga
