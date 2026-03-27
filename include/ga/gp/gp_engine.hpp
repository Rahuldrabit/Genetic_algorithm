#pragma once

#include "ga/gp/expression_tree.hpp"
#include <random>
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>
#include <algorithm>

namespace ga {
namespace gp {

// ============================================================================
// GPConfig – configuration for a GP run
// ============================================================================

struct GPConfig {
    int  populationSize  = 50;
    int  generations     = 50;
    int  maxTreeDepth    = 6;
    int  initMaxDepth    = 4;   ///< max depth used during initialisation
    double crossoverRate = 0.8;
    double mutationRate  = 0.1;
    double eliteRatio    = 0.05;
    unsigned seed        = 0;

    /// When true, type constraints are enforced (strongly-typed GP).
    /// When false, any primitive can be combined freely (loosely-typed GP).
    bool stronglyTyped   = false;

    GPType rootType      = GPType::Real;  ///< expected type of tree root
};

// ============================================================================
// GPResult – result of a GP run
// ============================================================================

struct GPResult {
    std::unique_ptr<ExprNode> bestTree;
    double bestFitness = -1e300;
    std::vector<double> bestHistory;
    std::vector<double> avgHistory;

    std::string bestProgram() const {
        return bestTree ? bestTree->toSExpr() : "";
    }
};

// ============================================================================
// GPEngine – runs Genetic Programming
//
// Supports both strongly-typed GP (cfg.stronglyTyped = true) and
// loosely-typed GP (cfg.stronglyTyped = false).
// ============================================================================

class GPEngine {
public:
    using FitnessFunc = std::function<double(const ExprNode&)>;

    GPEngine(GPConfig cfg, const PrimitiveSet& pset)
        : cfg_(std::move(cfg)), pset_(pset)
    {
        if (pset_.empty())
            throw std::invalid_argument("GPEngine: primitive set must not be empty");

        unsigned seed = cfg_.seed == 0
            ? std::random_device{}()
            : cfg_.seed;
        rng_.seed(seed);
    }

    GPResult run(const FitnessFunc& fitness);

private:
    // ---- individual ----
    struct Ind {
        std::unique_ptr<ExprNode> tree;
        double fit = -1e300;
    };

    GPConfig     cfg_;
    const PrimitiveSet& pset_;
    std::mt19937 rng_;

    // ---- initialisation ----
    std::unique_ptr<ExprNode> grow(int maxDepth, GPType targetType);
    std::unique_ptr<ExprNode> full(int depth,    GPType targetType);
    std::vector<Ind> initPopulation(const FitnessFunc& f);

    // ---- genetic operators ----
    std::unique_ptr<ExprNode> subtreeCrossover(const ExprNode& p1,
                                                const ExprNode& p2);
    void pointMutation(ExprNode& tree, int depth = 0);

    // ---- tree navigation helpers ----
    /// Collect raw pointers to all nodes
    void collectNodes(ExprNode& node, std::vector<ExprNode*>& out);

    /// Randomly pick a compatible replacement primitive
    const Primitive* pickReplacement(const ExprNode& node,
                                     const std::vector<const Primitive*>& cands,
                                     const std::vector<const Primitive*>& terms);

    // ---- selection (tournament) ----
    const Ind& tournamentSelect(const std::vector<Ind>& pop, int k = 3);
};

} // namespace gp
} // namespace ga
