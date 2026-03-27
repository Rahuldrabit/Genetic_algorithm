#ifndef POINT_MUTATION_H
#define POINT_MUTATION_H

#include "base_mutation.h"
#include "ga/gp/expression_tree.hpp"

/// Point mutation for GP expression trees.
///
/// Each node in the tree is independently replaced with a randomly chosen
/// primitive of the same arity (and, when strongly-typed, the same return
/// type).  This mirrors the DEAP 'mutNodeReplacement' operator.
class PointMutation : public MutationOperator {
public:
    explicit PointMutation(unsigned seed = std::random_device{}())
        : MutationOperator("PointMutation", seed) {}

    /**
     * @brief Apply point mutation to a GP expression tree.
     *
     * Each node is visited and, with probability @p pm, replaced by a
     * randomly sampled compatible primitive from @p pset.
     *
     * @param tree       Root of the expression tree to mutate (modified in place).
     * @param pset       Primitive set used for sampling replacements.
     * @param pm         Per-node mutation probability.
     * @param strongTyped  When true, only primitives with matching return type
     *                     and identical argument-type signature are considered.
     */
    void mutate(ga::gp::ExprNode& tree,
                const ga::gp::PrimitiveSet& pset,
                double pm,
                bool stronglyTyped = false) const;

private:
    void visitNode(ga::gp::ExprNode& node,
                   const ga::gp::PrimitiveSet& pset,
                   double pm,
                   bool stronglyTyped) const;
};

#endif // POINT_MUTATION_H
