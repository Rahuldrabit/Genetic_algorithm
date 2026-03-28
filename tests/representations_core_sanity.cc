/**
 * Sanity tests for new representation types and core abstractions.
 * Tests BitsetGenome, PermutationGenome, Population, IEvaluator, IAlgorithm,
 * core::OptimizationResult, and the OptimizerBuilder.
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

// New representations
#include "ga/representations/bitset_genome.hpp"
#include "ga/representations/permutation_genome.hpp"

// New core abstractions
#include "ga/core/population.hpp"
#include "ga/core/evaluator.hpp"
#include "ga/core/engine.hpp"
#include "ga/core/result.hpp"
#include "ga/core/individual.hpp"

// Builder API
#include "ga/api/builder.hpp"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] " << msg << "\n"; \
            ++tests_failed; \
            return false; \
        } \
    } while (0)

#define RUN_TEST(fn) \
    do { \
        if (fn()) { \
            std::cout << "[PASS] " << #fn << "\n"; \
            ++tests_passed; \
        } else { \
            ++tests_failed; \
        } \
    } while (0)

// ── BitsetGenome ─────────────────────────────────────────────────────────────

static bool test_bitset_genome_basic() {
    using namespace ga::representations;

    BitsetGenome empty;
    TEST_ASSERT(empty.bits.empty(), "Default BitsetGenome should be empty");

    BitsetGenome filled(8, false);
    TEST_ASSERT(filled.size() == 8, "BitsetGenome size mismatch");
    TEST_ASSERT(filled.popcount() == 0, "popcount of all-false should be 0");

    BitsetGenome ones(4, true);
    TEST_ASSERT(ones.popcount() == 4, "popcount of all-true(4) should be 4");

    BitsetGenome custom(std::vector<bool>{true, false, true, false, true});
    TEST_ASSERT(custom.popcount() == 3, "popcount of {1,0,1,0,1} should be 3");

    auto cloned = custom.clone();
    TEST_ASSERT(cloned != nullptr, "clone() returned null");
    TEST_ASSERT(cloned->encodingName() == "bitset", "encodingName mismatch");

    auto* cg = dynamic_cast<BitsetGenome*>(cloned.get());
    TEST_ASSERT(cg != nullptr, "clone dynamic_cast failed");
    TEST_ASSERT(cg->bits == custom.bits, "clone bits mismatch");

    return true;
}

static bool test_bitset_genome_hamming() {
    using namespace ga::representations;

    BitsetGenome a(std::vector<bool>{true, false, true, false});
    BitsetGenome b(std::vector<bool>{true, true, false, false});
    // differ at positions 1 and 2 → distance 2
    TEST_ASSERT(a.hammingDistance(b) == 2, "Hamming distance should be 2");

    BitsetGenome same(std::vector<bool>{true, false, true, false});
    TEST_ASSERT(a.hammingDistance(same) == 0, "Hamming distance to self-copy should be 0");

    return true;
}

// ── PermutationGenome ────────────────────────────────────────────────────────

static bool test_permutation_genome_basic() {
    using namespace ga::representations;

    PermutationGenome identity(5);
    TEST_ASSERT(identity.size() == 5, "identity size should be 5");
    for (int i = 0; i < 5; ++i) {
        TEST_ASSERT(identity.order[static_cast<std::size_t>(i)] == i, "identity order mismatch");
    }
    TEST_ASSERT(identity.isValid(), "identity permutation should be valid");

    PermutationGenome custom(std::vector<int>{3, 0, 4, 1, 2});
    TEST_ASSERT(custom.isValid(), "custom permutation should be valid");
    TEST_ASSERT(custom.positionOf(4) == 2, "positionOf(4) should be 2");

    PermutationGenome invalid(std::vector<int>{0, 0, 2});
    TEST_ASSERT(!invalid.isValid(), "duplicate-entry permutation should be invalid");

    return true;
}

static bool test_permutation_genome_random() {
    using namespace ga::representations;

    std::mt19937 rng(123);
    auto perm = PermutationGenome::random(10, rng);
    TEST_ASSERT(perm.size() == 10, "random perm size mismatch");
    TEST_ASSERT(perm.isValid(), "random permutation should be valid");

    // Check that all values [0..9] are present.
    std::vector<int> sorted = perm.order;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < 10; ++i) {
        TEST_ASSERT(sorted[static_cast<std::size_t>(i)] == i, "random perm missing value");
    }

    return true;
}

static bool test_permutation_genome_clone() {
    using namespace ga::representations;

    PermutationGenome original(std::vector<int>{2, 0, 1});
    auto cloned = original.clone();
    TEST_ASSERT(cloned != nullptr, "clone() returned null");
    TEST_ASSERT(cloned->encodingName() == "permutation", "encodingName mismatch");

    auto* pg = dynamic_cast<PermutationGenome*>(cloned.get());
    TEST_ASSERT(pg != nullptr, "clone dynamic_cast failed");
    TEST_ASSERT(pg->order == original.order, "cloned order mismatch");

    return true;
}

// ── Population utilities ─────────────────────────────────────────────────────

static bool test_population_utilities() {
    using namespace ga::core;

    Population pop(4);
    pop[0].evaluation.objectives = {3.0};
    pop[1].evaluation.objectives = {1.0};
    pop[2].evaluation.objectives = {4.0};
    pop[3].evaluation.objectives = {2.0};

    std::size_t best = bestIndex(pop);
    TEST_ASSERT(best == 1, "bestIndex should return index 1 (objective 1.0)");

    Population top2 = topK(pop, 2);
    TEST_ASSERT(top2.size() == 2, "topK(2) should return 2 individuals");
    TEST_ASSERT(top2[0].evaluation.objectives[0] == 1.0, "topK first should have objective 1.0");
    TEST_ASSERT(top2[1].evaluation.objectives[0] == 2.0, "topK second should have objective 2.0");

    Population top10 = topK(pop, 10);
    TEST_ASSERT(top10.size() == 4, "topK clamps to population size");

    return true;
}

// ── IEvaluator interface ─────────────────────────────────────────────────────

static bool test_evaluator_interface() {
    using namespace ga::core;
    using namespace ga::representations;

    // Concrete evaluator: sum of squared genes for a VectorGenome<double>.
    struct SphereEvaluator : IEvaluator {
        ga::Evaluation evaluate(const ga::IGenome& genome) const override {
            const auto* vg = dynamic_cast<const VectorGenome<double>*>(&genome);
            if (!vg) {
                throw std::runtime_error("SphereEvaluator: expected VectorGenome<double>");
            }
            double sum = 0.0;
            for (double g : vg->genes) {
                sum += g * g;
            }
            return {{sum}, true, 0.0};
        }
    };

    SphereEvaluator eval;
    VectorGenome<double> g1(std::vector<double>{3.0, 4.0});
    auto result = eval.evaluate(g1);
    TEST_ASSERT(std::fabs(result.objectives[0] - 25.0) < 1e-9, "sphere eval should be 25.0");
    TEST_ASSERT(result.feasible, "feasible should be true");

    // Test batch evaluation
    VectorGenome<double> g2(std::vector<double>{1.0, 0.0});
    std::vector<const ga::IGenome*> batch{&g1, &g2};
    auto batchResults = eval.evaluateBatch(batch);
    TEST_ASSERT(batchResults.size() == 2, "batch size mismatch");
    TEST_ASSERT(std::fabs(batchResults[0].objectives[0] - 25.0) < 1e-9, "batch[0] sphere mismatch");
    TEST_ASSERT(std::fabs(batchResults[1].objectives[0] - 1.0) < 1e-9, "batch[1] sphere mismatch");

    return true;
}

// ── IAlgorithm / engine interface ────────────────────────────────────────────

static bool test_engine_interface() {
    using namespace ga::core;
    using namespace ga::representations;

    // Toy algorithm: each step adds one Individual to the population.
    struct ToyAlgorithm : IAlgorithm {
        Population pop_;
        std::size_t gen_ = 0;

        void step() override {
            ga::Individual ind;
            ind.genome = std::make_unique<VectorGenome<double>>(std::vector<double>{static_cast<double>(gen_)});
            ind.evaluation.objectives = {static_cast<double>(gen_)};
            pop_.push_back(std::move(ind));
            ++gen_;
        }

        const Population& population() const override { return pop_; }
        std::size_t generation() const override { return gen_; }
    };

    ToyAlgorithm algo;
    TEST_ASSERT(algo.generation() == 0, "initial generation should be 0");

    algo.run(5);
    TEST_ASSERT(algo.generation() == 5, "generation after run(5) should be 5");
    TEST_ASSERT(algo.population().size() == 5, "population size after run(5) should be 5");

    return true;
}

// ── OptimizationResult ───────────────────────────────────────────────────────

static bool test_optimization_result() {
    using namespace ga::core;

    OptimizationResult r;
    TEST_ASSERT(r.bestFitness == 0.0, "default bestFitness");
    TEST_ASSERT(r.evaluations == 0, "default evaluations");

    r.bestSolution = {1.0, 2.0};
    r.bestFitness = 42.0;
    r.bestHistory = {10.0, 20.0, 42.0};
    r.evaluations = 300;
    r.generations = 3;

    TEST_ASSERT(r.bestSolution.size() == 2, "bestSolution size");
    TEST_ASSERT(r.bestFitness == 42.0, "bestFitness");
    TEST_ASSERT(r.evaluations == 300, "evaluations");

    return true;
}

// ── OptimizerBuilder ─────────────────────────────────────────────────────────

static bool test_optimizer_builder() {
    auto optimizer = ga::api::OptimizerBuilder()
        .dimension(5)
        .bounds(-2.0, 2.0)
        .populationSize(30)
        .generations(20)
        .seed(7)
        .threads(1)
        .build();

    auto result = optimizer.optimize([](const std::vector<double>& x) {
        double s = 0.0;
        for (double v : x) s += v * v;
        return 1000.0 / (1.0 + s);
    });

    TEST_ASSERT(!result.bestGenes.empty(), "OptimizerBuilder result should not be empty");
    TEST_ASSERT(result.bestFitness > 0.0, "OptimizerBuilder bestFitness should be positive");

    return true;
}

static bool test_optimizer_builder_validation() {
    try {
        ga::api::OptimizerBuilder().bounds(5.0, -1.0);
        return false; // should have thrown
    } catch (const std::invalid_argument&) {
        // expected
    }

    try {
        ga::api::OptimizerBuilder().populationSize(-1);
        return false;
    } catch (const std::invalid_argument&) {
        // expected
    }

    try {
        ga::api::OptimizerBuilder().mutationRate(1.5);
        return false;
    } catch (const std::invalid_argument&) {
        // expected
    }

    return true;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "========================================\n";
    std::cout << "  New Representations & Core Abstractions Sanity Tests\n";
    std::cout << "========================================\n\n";

    RUN_TEST(test_bitset_genome_basic);
    RUN_TEST(test_bitset_genome_hamming);
    RUN_TEST(test_permutation_genome_basic);
    RUN_TEST(test_permutation_genome_random);
    RUN_TEST(test_permutation_genome_clone);
    RUN_TEST(test_population_utilities);
    RUN_TEST(test_evaluator_interface);
    RUN_TEST(test_engine_interface);
    RUN_TEST(test_optimization_result);
    RUN_TEST(test_optimizer_builder);
    RUN_TEST(test_optimizer_builder_validation);

    std::cout << "\n========================================\n";
    std::cout << "  Test Summary\n";
    std::cout << "========================================\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n\n";

    if (tests_failed == 0) {
        std::cout << "ALL PASS\n";
    } else {
        std::cout << "SOME TESTS FAILED\n";
    }

    return tests_failed == 0 ? 0 : 1;
}
