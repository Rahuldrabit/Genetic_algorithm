#include "point_mutation.h"

#include <vector>

using namespace ga::gp;

// ============================================================================
// POINT MUTATION IMPLEMENTATION
// ============================================================================

void PointMutation::visitNode(ExprNode& node,
                              const PrimitiveSet& pset,
                              double pm,
                              bool stronglyTyped) const
{
    if (uniform_dist(rng) < pm) {
        stats.totalMutations++;

        if (node.isLeaf()) {
            // Replace terminal
            std::vector<const Primitive*> cands;
            for (const auto& p : pset.terminals())
                if (!stronglyTyped || p.returnType == node.type)
                    cands.push_back(&p);

            if (!cands.empty()) {
                std::uniform_int_distribution<size_t> d(0, cands.size() - 1);
                const Primitive* chosen = cands[d(rng)];
                node.value = chosen->name;
                node.type  = chosen->returnType;
                stats.successfulMutations++;
            }
        } else {
            // Replace function — keep the same arity
            std::vector<const Primitive*> cands;
            for (const auto& p : pset.functions())
                if ((int)p.argTypes.size() == node.arity() &&
                    (!stronglyTyped || p.returnType == node.type))
                    cands.push_back(&p);

            if (!cands.empty()) {
                std::uniform_int_distribution<size_t> d(0, cands.size() - 1);
                const Primitive* chosen = cands[d(rng)];
                node.value = chosen->name;
                node.type  = chosen->returnType;
                // Update child types when strongly typed
                if (stronglyTyped) {
                    for (int i = 0; i < node.arity(); ++i)
                        node.children[i]->type = chosen->argTypes[i];
                }
                stats.successfulMutations++;
            }
        }
    }

    // Recurse into children
    for (auto& ch : node.children)
        visitNode(*ch, pset, pm, stronglyTyped);
}

void PointMutation::mutate(ExprNode& tree,
                           const PrimitiveSet& pset,
                           double pm,
                           bool stronglyTyped) const
{
    validateProbability(pm, "pointMutation");
    visitNode(tree, pset, pm, stronglyTyped);
}
