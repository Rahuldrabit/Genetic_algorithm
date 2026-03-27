#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <stdexcept>
#include <cmath>

namespace ga {
namespace gp {

// ============================================================================
// GP Type system (used by both strongly-typed and loosely-typed GP)
// ============================================================================

/// Primitive type tags for strongly-typed GP
enum class GPType {
    Real,    ///< floating-point value
    Bool,    ///< boolean (used as a guard / predicate)
    Int,     ///< integer
    Any      ///< wildcard — used by loosely-typed GP
};

inline const char* gpTypeName(GPType t) {
    switch (t) {
        case GPType::Real: return "Real";
        case GPType::Bool: return "Bool";
        case GPType::Int:  return "Int";
        case GPType::Any:  return "Any";
    }
    return "Unknown";
}

// ============================================================================
// ExprNode – a single node in a GP expression tree
//
// Prefix-tree layout:  (+ (* x y) 3)
//   value = "+"
//   children[0].value = "*", children[0].children = {x, y}
//   children[1].value = "3"   (leaf / terminal)
// ============================================================================

struct ExprNode {
    std::string            value;      ///< operator name or terminal value
    GPType                 type;       ///< return type of this sub-expression
    std::vector<std::unique_ptr<ExprNode>> children;

    // ---- construction helpers ----
    explicit ExprNode(std::string val, GPType t = GPType::Any)
        : value(std::move(val)), type(t) {}

    bool isLeaf() const { return children.empty(); }
    int  arity()  const { return static_cast<int>(children.size()); }

    // ---- deep clone ----
    std::unique_ptr<ExprNode> clone() const {
        auto copy = std::make_unique<ExprNode>(value, type);
        for (const auto& ch : children)
            copy->children.push_back(ch->clone());
        return copy;
    }

    // ---- S-expression string ----
    std::string toSExpr() const {
        if (isLeaf()) return value;
        std::string s = '(' + value;
        for (const auto& ch : children)
            s += ' ' + ch->toSExpr();
        return s + ')';
    }

    // ---- node count ----
    int size() const {
        int n = 1;
        for (const auto& ch : children) n += ch->size();
        return n;
    }

    // ---- depth ----
    int depth() const {
        if (isLeaf()) return 0;
        int d = 0;
        for (const auto& ch : children) d = std::max(d, ch->depth());
        return 1 + d;
    }
};

// ============================================================================
// Primitive – descriptor for a function or terminal used in the primitive set
// ============================================================================

struct Primitive {
    std::string name;
    GPType      returnType;
    std::vector<GPType> argTypes;   ///< empty for terminals
    /// Evaluation function: receives evaluated children as doubles
    std::function<double(const std::vector<double>&)> eval;

    bool isTerminal() const { return argTypes.empty(); }
    int  arity()      const { return static_cast<int>(argTypes.size()); }
};

// ============================================================================
// PrimitiveSet – collection of functions and terminals for a GP problem
// ============================================================================

class PrimitiveSet {
public:
    /// Add a function primitive (operator/node with children)
    void addFunction(const std::string& name,
                     GPType returnType,
                     std::vector<GPType> argTypes,
                     std::function<double(const std::vector<double>&)> eval)
    {
        Primitive p;
        p.name = name;
        p.returnType = returnType;
        p.argTypes = std::move(argTypes);
        p.eval = std::move(eval);
        functions_.push_back(std::move(p));
    }

    /// Add a terminal primitive (leaf: variable or ephemeral constant)
    void addTerminal(const std::string& name,
                     GPType returnType,
                     std::function<double(const std::vector<double>&)> eval)
    {
        Primitive p;
        p.name = name;
        p.returnType = returnType;
        p.eval = std::move(eval);
        terminals_.push_back(std::move(p));
    }

    const std::vector<Primitive>& functions() const { return functions_; }
    const std::vector<Primitive>& terminals() const { return terminals_; }

    /// Functions that return a given type (for strongly-typed GP)
    std::vector<const Primitive*> functionsOfType(GPType t) const {
        std::vector<const Primitive*> res;
        for (const auto& p : functions_)
            if (t == GPType::Any || p.returnType == t) res.push_back(&p);
        return res;
    }

    /// Terminals that return a given type
    std::vector<const Primitive*> terminalsOfType(GPType t) const {
        std::vector<const Primitive*> res;
        for (const auto& p : terminals_)
            if (t == GPType::Any || p.returnType == t) res.push_back(&p);
        return res;
    }

    bool empty() const { return functions_.empty() && terminals_.empty(); }

    // ---- Convenience factory: standard arithmetic primitives ----
    static PrimitiveSet makeArithmetic(const std::vector<std::string>& varNames) {
        PrimitiveSet ps;
        // Functions
        ps.addFunction("+", GPType::Real, {GPType::Real, GPType::Real},
            [](const std::vector<double>& a){ return a[0] + a[1]; });
        ps.addFunction("-", GPType::Real, {GPType::Real, GPType::Real},
            [](const std::vector<double>& a){ return a[0] - a[1]; });
        ps.addFunction("*", GPType::Real, {GPType::Real, GPType::Real},
            [](const std::vector<double>& a){ return a[0] * a[1]; });
        // Protected division
        ps.addFunction("/", GPType::Real, {GPType::Real, GPType::Real},
            [](const std::vector<double>& a){
                return std::abs(a[1]) < 1e-10 ? 1.0 : a[0] / a[1];
            });
        ps.addFunction("sin", GPType::Real, {GPType::Real},
            [](const std::vector<double>& a){ return std::sin(a[0]); });
        ps.addFunction("cos", GPType::Real, {GPType::Real},
            [](const std::vector<double>& a){ return std::cos(a[0]); });
        // Terminals: variables
        for (size_t i = 0; i < varNames.size(); ++i) {
            const size_t idx = i;
            ps.addTerminal(varNames[i], GPType::Real,
                [idx](const std::vector<double>& vars) -> double { return vars[idx]; });
        }
        // Ephemeral constant 1.0
        ps.addTerminal("1", GPType::Real,
            [](const std::vector<double>&){ return 1.0; });
        return ps;
    }

private:
    std::vector<Primitive> functions_;
    std::vector<Primitive> terminals_;
};

// ============================================================================
// Tree evaluation
// ============================================================================

/// Evaluate an expression tree given a mapping from variable name to index and
/// the input variable vector.
double evaluate(const ExprNode& node,
                const PrimitiveSet& pset,
                const std::vector<double>& vars);

// ============================================================================
// TypeChecker – validates a tree against strongly-typed GP rules
// ============================================================================

class TypeChecker {
public:
    explicit TypeChecker(const PrimitiveSet& pset) : pset_(pset) {}

    /// Returns true if every node's return type matches the expected argument
    /// type of its parent (strong typing).  Loosely-typed trees (GPType::Any)
    /// are always accepted.
    bool check(const ExprNode& node, GPType expectedType = GPType::Any) const {
        if (expectedType != GPType::Any &&
            node.type    != GPType::Any &&
            node.type    != expectedType)
            return false;

        // Find primitive to get argument types
        for (const auto& p : pset_.functions()) {
            if (p.name == node.value && (int)p.argTypes.size() == node.arity()) {
                for (int i = 0; i < node.arity(); ++i)
                    if (!check(*node.children[i], p.argTypes[i]))
                        return false;
                return true;
            }
        }
        for (const auto& p : pset_.terminals()) {
            if (p.name == node.value && node.isLeaf())
                return true;
        }
        // Unknown node (e.g., numeric constant) – accept in loose mode
        return node.isLeaf();
    }

private:
    const PrimitiveSet& pset_;
};

} // namespace gp
} // namespace ga
