#pragma once
/// Automatically Defined Functions (ADF)
///
/// ADFs are reusable sub-programs evolved alongside the main program tree.
/// During a GP run each individual can be augmented with an ADF pool; the
/// main tree may call any ADF by name, treating it as an extra terminal /
/// zero-arity function that returns a Real value.
///
/// Usage:
///   ADFPool pool(2 /*num_adfs*/, pset);
///   pool.evolveADFs(…);          // optional: pre-evolve ADFs
///   pool.registerInto(pset_main);// inject ADF callables into the main pset
///   // now run GPEngine with pset_main – it will call "ADF0", "ADF1", …

#include "ga/gp/expression_tree.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>

namespace ga {
namespace gp {

// ============================================================================
// ADFDefinition – one named, evolved sub-function
// ============================================================================

struct ADFDefinition {
    std::string name;                          ///< e.g. "ADF0"
    std::unique_ptr<ExprNode> body;            ///< expression tree body
    std::vector<std::string>  argNames;        ///< formal argument names (terminals inside body)
    GPType                    returnType = GPType::Real;

    /// Deep clone
    ADFDefinition clone() const {
        ADFDefinition c;
        c.name       = name;
        c.body       = body ? body->clone() : nullptr;
        c.argNames   = argNames;
        c.returnType = returnType;
        return c;
    }

    /// Evaluate the ADF body given concrete argument values.
    /// @p pset    Primitive set used to evaluate the body tree.
    /// @p argVals Values for each formal argument (same order as argNames).
    double evaluate(const PrimitiveSet& pset,
                    const std::vector<double>& argVals) const;
};

// ============================================================================
// ADFPool – collection of ADFs attached to one individual
// ============================================================================

class ADFPool {
public:
    explicit ADFPool() = default;

    /// Add an ADF definition.
    void addADF(ADFDefinition adf) {
        adfs_.push_back(std::move(adf));
    }

    /// Return number of ADFs.
    size_t size() const { return adfs_.size(); }
    bool   empty() const { return adfs_.empty(); }

    const ADFDefinition& operator[](size_t i) const { return adfs_[i]; }
          ADFDefinition& operator[](size_t i)       { return adfs_[i]; }

    const std::vector<ADFDefinition>& adfs() const { return adfs_; }

    /// Deep clone this pool (used when cloning individuals).
    ADFPool clone() const {
        ADFPool copy;
        for (const auto& a : adfs_) copy.adfs_.push_back(a.clone());
        return copy;
    }

    /// Register each ADF as a zero-arity terminal in @p target so that the
    /// main GP tree can call "ADF0", "ADF1", … as ordinary terminal symbols.
    ///
    /// @p mainPset  The primitive set used for the main tree evaluation.
    ///              Shared (by const ref) – we capture a pointer to @p pool.
    void registerInto(PrimitiveSet& target,
                      const PrimitiveSet& bodyPset) const {
        for (size_t i = 0; i < adfs_.size(); ++i) {
            const ADFDefinition* adf = &adfs_[i];
            const PrimitiveSet*  bp  = &bodyPset;
            target.addTerminal(adf->name, adf->returnType,
                [adf, bp](const std::vector<double>& vars) -> double {
                    // Call the ADF with the same variable bindings
                    return adf->evaluate(*bp, vars);
                });
        }
    }

private:
    std::vector<ADFDefinition> adfs_;
};

} // namespace gp
} // namespace ga
