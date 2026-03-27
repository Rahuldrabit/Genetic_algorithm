#pragma once

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <set>
#include <unordered_map>
#include <functional>
#include <stdexcept>

namespace ga {

// ============================================================================
// Base Individual – pure-virtual fitness, polymorphic cloning
// ============================================================================

class Individual {
public:
    virtual ~Individual() = default;

    /// Evaluate and return the individual's fitness (higher = better).
    virtual double fitness() const = 0;

    /// Deep-copy this individual.
    virtual std::unique_ptr<Individual> clone() const = 0;

    /// Human-readable description.
    virtual std::string toString() const = 0;
};

// ============================================================================
// VectorIndividual – real-valued gene vector
// (covers List / Array / Numpy-like use-cases)
// ============================================================================

class VectorIndividual : public Individual {
public:
    using FitnessFunc = std::function<double(const std::vector<double>&)>;

    explicit VectorIndividual(std::vector<double> genes, FitnessFunc fn)
        : genes_(std::move(genes)), fn_(std::move(fn)) {}

    double fitness() const override { return fn_(genes_); }

    std::unique_ptr<Individual> clone() const override {
        return std::make_unique<VectorIndividual>(genes_, fn_);
    }

    std::string toString() const override {
        std::string s = "VectorIndividual[";
        for (size_t i = 0; i < genes_.size(); ++i) {
            if (i) s += ',';
            s += std::to_string(genes_[i]);
        }
        return s + "]";
    }

    std::vector<double>&       genes()       { return genes_; }
    const std::vector<double>& genes() const { return genes_; }

private:
    std::vector<double> genes_;
    FitnessFunc fn_;
};

// ============================================================================
// SetIndividual – chromosome represented as an ordered set of integers
// ============================================================================

class SetIndividual : public Individual {
public:
    using FitnessFunc = std::function<double(const std::set<int>&)>;

    explicit SetIndividual(std::set<int> genes, FitnessFunc fn)
        : genes_(std::move(genes)), fn_(std::move(fn)) {}

    double fitness() const override { return fn_(genes_); }

    std::unique_ptr<Individual> clone() const override {
        return std::make_unique<SetIndividual>(genes_, fn_);
    }

    std::string toString() const override {
        std::string s = "SetIndividual{";
        bool first = true;
        for (int v : genes_) {
            if (!first) s += ',';
            s += std::to_string(v);
            first = false;
        }
        return s + "}";
    }

    std::set<int>&       genes()       { return genes_; }
    const std::set<int>& genes() const { return genes_; }

private:
    std::set<int> genes_;
    FitnessFunc fn_;
};

// ============================================================================
// DictIndividual – chromosome as a string-to-double dictionary
// ============================================================================

class DictIndividual : public Individual {
public:
    using GeneMap    = std::unordered_map<std::string, double>;
    using FitnessFunc = std::function<double(const GeneMap&)>;

    explicit DictIndividual(GeneMap genes, FitnessFunc fn)
        : genes_(std::move(genes)), fn_(std::move(fn)) {}

    double fitness() const override { return fn_(genes_); }

    std::unique_ptr<Individual> clone() const override {
        return std::make_unique<DictIndividual>(genes_, fn_);
    }

    std::string toString() const override {
        std::string s = "DictIndividual{";
        bool first = true;
        for (const auto& kv : genes_) {
            if (!first) s += ',';
            s += kv.first + ':' + std::to_string(kv.second);
            first = false;
        }
        return s + "}";
    }

    GeneMap&       genes()       { return genes_; }
    const GeneMap& genes() const { return genes_; }

private:
    GeneMap genes_;
    FitnessFunc fn_;
};

// ============================================================================
// TreeIndividual – chromosome is a GP expression tree
// (forward-declared; full type in gp/expression_tree.hpp)
// ============================================================================

namespace gp { struct ExprNode; }

class TreeIndividual : public Individual {
public:
    using NodePtr     = std::unique_ptr<gp::ExprNode>;
    using FitnessFunc = std::function<double(const gp::ExprNode&)>;

    explicit TreeIndividual(NodePtr root, FitnessFunc fn)
        : root_(std::move(root)), fn_(std::move(fn)) {}

    double fitness() const override;

    std::unique_ptr<Individual> clone() const override;

    std::string toString() const override;

    gp::ExprNode*       root()       { return root_.get(); }
    const gp::ExprNode* root() const { return root_.get(); }

    /// Transfer ownership of the root node.
    NodePtr releaseRoot() { return std::move(root_); }

    /// Replace the root (used after crossover / mutation).
    void setRoot(NodePtr r) { root_ = std::move(r); }

    const FitnessFunc& fitnessFunc() const { return fn_; }

private:
    NodePtr root_;
    FitnessFunc fn_;
};

// ============================================================================
// GraphIndividual – chromosome as an adjacency list (directed graph)
// Useful for neural-architecture search, routing, network design, etc.
// ============================================================================

class GraphIndividual : public Individual {
public:
    using AdjList     = std::vector<std::vector<int>>;
    using FitnessFunc = std::function<double(const AdjList&)>;

    explicit GraphIndividual(AdjList adj, FitnessFunc fn)
        : adj_(std::move(adj)), fn_(std::move(fn)) {}

    double fitness() const override { return fn_(adj_); }

    std::unique_ptr<Individual> clone() const override {
        return std::make_unique<GraphIndividual>(adj_, fn_);
    }

    std::string toString() const override {
        std::string s = "GraphIndividual(nodes=" + std::to_string(adj_.size()) + ")";
        return s;
    }

    AdjList&       adj()       { return adj_; }
    const AdjList& adj() const { return adj_; }

private:
    AdjList adj_;
    FitnessFunc fn_;
};

} // namespace ga
