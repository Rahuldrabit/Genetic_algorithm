/// gp_sanity.cc – lightweight sanity tests for the GP features:
///   1. ExprNode construction, clone, toSExpr, size, depth
///   2. PrimitiveSet factory and evaluation
///   3. TypeChecker (strong typing)
///   4. VectorIndividual / SetIndividual / DictIndividual / TreeIndividual /
///      GraphIndividual  (Individual hierarchy)
///   5. GPEngine (loosely-typed and strongly-typed) – does it return a tree?
///   6. PointMutation (GP)

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <set>
#include <unordered_map>

#include "ga/individual.hpp"
#include "ga/gp/expression_tree.hpp"
#include "ga/gp/gp_engine.hpp"
#include "mutation/point_mutation.h"

using namespace ga;
using namespace ga::gp;

static void pass(const std::string& name) { std::cout << "[PASS] " << name << '\n'; }
static void fail(const std::string& name, const std::string& why) {
    std::cout << "[FAIL] " << name << " - " << why << '\n';
}

// ---------------------------------------------------------------------------
// 1. ExprNode basics
// ---------------------------------------------------------------------------
static bool test_expr_node() {
    // Build  (+ (* x y) 1)
    auto root = std::make_unique<ExprNode>("+", GPType::Real);
    {
        auto mul = std::make_unique<ExprNode>("*", GPType::Real);
        mul->children.push_back(std::make_unique<ExprNode>("x", GPType::Real));
        mul->children.push_back(std::make_unique<ExprNode>("y", GPType::Real));
        root->children.push_back(std::move(mul));
    }
    root->children.push_back(std::make_unique<ExprNode>("1", GPType::Real));

    if (root->size() != 5) { fail("ExprNode::size", "expected 5"); return false; }
    if (root->depth() != 2) { fail("ExprNode::depth", "expected 2"); return false; }
    std::string s = root->toSExpr();
    if (s != "(+ (* x y) 1)") { fail("ExprNode::toSExpr", "got: " + s); return false; }

    // Clone
    auto clone = root->clone();
    if (clone->toSExpr() != s) { fail("ExprNode::clone", "mismatch"); return false; }

    pass("ExprNode basics");
    return true;
}

// ---------------------------------------------------------------------------
// 2. PrimitiveSet + evaluate
// ---------------------------------------------------------------------------
static bool test_evaluate() {
    auto pset = PrimitiveSet::makeArithmetic({"x", "y"});

    // Build (+ x 1) — expected value with x=3: 4
    auto root = std::make_unique<ExprNode>("+", GPType::Real);
    root->children.push_back(std::make_unique<ExprNode>("x", GPType::Real));
    root->children.push_back(std::make_unique<ExprNode>("1", GPType::Real));

    std::vector<double> vars = {3.0, 0.0};
    double val = evaluate(*root, pset, vars);
    if (std::abs(val - 4.0) > 1e-9) { fail("evaluate (+ x 1)", "expected 4, got " + std::to_string(val)); return false; }

    // Build (* x y) — expected with x=2, y=5: 10
    auto mul = std::make_unique<ExprNode>("*", GPType::Real);
    mul->children.push_back(std::make_unique<ExprNode>("x", GPType::Real));
    mul->children.push_back(std::make_unique<ExprNode>("y", GPType::Real));
    vars = {2.0, 5.0};
    double v2 = evaluate(*mul, pset, vars);
    if (std::abs(v2 - 10.0) > 1e-9) { fail("evaluate (* x y)", "expected 10, got " + std::to_string(v2)); return false; }

    // Protected division by zero → 1
    auto divz = std::make_unique<ExprNode>("/", GPType::Real);
    divz->children.push_back(std::make_unique<ExprNode>("x", GPType::Real));
    divz->children.push_back(std::make_unique<ExprNode>("y", GPType::Real));
    vars = {5.0, 0.0};
    double v3 = evaluate(*divz, pset, vars);
    if (std::abs(v3 - 1.0) > 1e-9) { fail("evaluate protected /", "expected 1, got " + std::to_string(v3)); return false; }

    pass("PrimitiveSet + evaluate");
    return true;
}

// ---------------------------------------------------------------------------
// 3. TypeChecker – strongly-typed validation
// ---------------------------------------------------------------------------
static bool test_type_checker() {
    auto pset = PrimitiveSet::makeArithmetic({"x"});
    TypeChecker tc(pset);

    // Valid: (+ x 1)
    auto good = std::make_unique<ExprNode>("+", GPType::Real);
    good->children.push_back(std::make_unique<ExprNode>("x", GPType::Real));
    good->children.push_back(std::make_unique<ExprNode>("1", GPType::Real));
    if (!tc.check(*good, GPType::Real)) { fail("TypeChecker valid tree", "rejected valid"); return false; }

    // TypeChecker with GPType::Any always passes
    auto any_node = std::make_unique<ExprNode>("x", GPType::Any);
    if (!tc.check(*any_node, GPType::Any)) { fail("TypeChecker Any", "rejected Any node"); return false; }

    pass("TypeChecker");
    return true;
}

// ---------------------------------------------------------------------------
// 4. Individual hierarchy
// ---------------------------------------------------------------------------
static bool test_individuals() {
    // VectorIndividual
    {
        VectorIndividual vi({1.0, 2.0, 3.0},
            [](const std::vector<double>& g){ return g[0]+g[1]+g[2]; });
        if (std::abs(vi.fitness() - 6.0) > 1e-9) { fail("VectorIndividual::fitness", ""); return false; }
        auto c = vi.clone();
        if (std::abs(c->fitness() - 6.0) > 1e-9) { fail("VectorIndividual::clone", ""); return false; }
    }

    // SetIndividual
    {
        SetIndividual si({1, 2, 3},
            [](const std::set<int>& s){ double r=0; for(int v:s) r+=v; return r; });
        if (std::abs(si.fitness() - 6.0) > 1e-9) { fail("SetIndividual::fitness", ""); return false; }
    }

    // DictIndividual
    {
        DictIndividual::GeneMap gm = {{"a", 2.0}, {"b", 3.0}};
        DictIndividual di(gm,
            [](const DictIndividual::GeneMap& m){ return m.at("a") * m.at("b"); });
        if (std::abs(di.fitness() - 6.0) > 1e-9) { fail("DictIndividual::fitness", ""); return false; }
    }

    // GraphIndividual
    {
        GraphIndividual::AdjList adj = {{1,2},{2},{}}; // 3 nodes
        GraphIndividual gi(adj,
            [](const GraphIndividual::AdjList& a){ return (double)a.size(); });
        if (std::abs(gi.fitness() - 3.0) > 1e-9) { fail("GraphIndividual::fitness", ""); return false; }
        auto c = gi.clone();
        if (std::abs(c->fitness() - 3.0) > 1e-9) { fail("GraphIndividual::clone", ""); return false; }
    }

    // TreeIndividual
    {
        auto pset = PrimitiveSet::makeArithmetic({"x"});
        auto root = std::make_unique<ExprNode>("x", GPType::Real);
        TreeIndividual ti(std::move(root),
            [&pset](const ExprNode& e){ return evaluate(e, pset, {5.0}); });
        if (std::abs(ti.fitness() - 5.0) > 1e-9) { fail("TreeIndividual::fitness", ""); return false; }
        auto c = ti.clone();
        if (std::abs(c->fitness() - 5.0) > 1e-9) { fail("TreeIndividual::clone", ""); return false; }
    }

    pass("Individual hierarchy (Vector, Set, Dict, Graph, Tree)");
    return true;
}

// ---------------------------------------------------------------------------
// 5. GPEngine – loosely-typed and strongly-typed
// ---------------------------------------------------------------------------
static bool test_gp_engine(bool stronglyTyped) {
    std::string label = stronglyTyped ? "GPEngine(strongly-typed)" : "GPEngine(loosely-typed)";

    auto pset = PrimitiveSet::makeArithmetic({"x"});

    GPConfig cfg;
    cfg.populationSize = 20;
    cfg.generations    = 10;
    cfg.maxTreeDepth   = 4;
    cfg.initMaxDepth   = 3;
    cfg.crossoverRate  = 0.8;
    cfg.mutationRate   = 0.3;
    cfg.eliteRatio     = 0.1;
    cfg.seed           = 42;
    cfg.stronglyTyped  = stronglyTyped;
    cfg.rootType       = GPType::Real;

    // Fitness: minimise |f(x) - x^2| for x in {1..5}
    // We reward closeness to x^2 by: 1 / (1 + RMSE)
    std::vector<double> xs = {1.0, 2.0, 3.0, 4.0, 5.0};
    GPEngine::FitnessFunc fit = [&pset, &xs](const ExprNode& tree) -> double {
        double err = 0.0;
        for (double xv : xs) {
            double predicted = evaluate(tree, pset, {xv});
            double target    = xv * xv;
            double d         = predicted - target;
            err += d * d;
        }
        return 1.0 / (1.0 + std::sqrt(err / xs.size()));
    };

    GPEngine engine(cfg, pset);
    GPResult res = engine.run(fit);

    if (!res.bestTree) { fail(label, "bestTree is null"); return false; }
    if (res.bestFitness <= 0.0) { fail(label, "bestFitness <= 0"); return false; }
    if ((int)res.bestHistory.size() != cfg.generations + 1) {
        fail(label, "bestHistory length mismatch (got " +
             std::to_string(res.bestHistory.size()) + ", expected " +
             std::to_string(cfg.generations + 1) + ")");
        return false;
    }

    pass(label);
    return true;
}

// ---------------------------------------------------------------------------
// 6. PointMutation
// ---------------------------------------------------------------------------
static bool test_point_mutation() {
    auto pset = PrimitiveSet::makeArithmetic({"x", "y"});

    // Build (+ x y)
    auto root = std::make_unique<ExprNode>("+", GPType::Real);
    root->children.push_back(std::make_unique<ExprNode>("x", GPType::Real));
    root->children.push_back(std::make_unique<ExprNode>("y", GPType::Real));

    std::string before = root->toSExpr();
    PointMutation pm(42);
    // High mutation probability to ensure something changes
    pm.mutate(*root, pset, 0.99, false);

    // Tree structure must still be valid (size == 3)
    if (root->size() != 3) { fail("PointMutation size preserved", ""); return false; }

    pass("PointMutation");
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    bool ok = true;
    ok &= test_expr_node();
    ok &= test_evaluate();
    ok &= test_type_checker();
    ok &= test_individuals();
    ok &= test_gp_engine(false);
    ok &= test_gp_engine(true);
    ok &= test_point_mutation();

    std::cout << (ok ? "ALL PASS" : "SOME FAILURES") << '\n';
    return ok ? 0 : 1;
}
