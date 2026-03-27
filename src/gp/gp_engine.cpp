#include "ga/gp/gp_engine.hpp"

#include <algorithm>
#include <numeric>
#include <cassert>
#include <cmath>

namespace ga {
namespace gp {

// ============================================================================
// Tree building
// ============================================================================

std::unique_ptr<ExprNode> GPEngine::grow(int maxDepth, GPType targetType) {
    // Candidates matching targetType
    auto fns  = cfg_.stronglyTyped ? pset_.functionsOfType(targetType)
                                   : pset_.functionsOfType(GPType::Any);
    auto terms = cfg_.stronglyTyped ? pset_.terminalsOfType(targetType)
                                    : pset_.terminalsOfType(GPType::Any);

    if (fns.empty() && terms.empty())
        throw std::runtime_error("GPEngine::grow: no primitives for type");

    // At max depth (or if no functions) → always pick a terminal
    if (maxDepth == 0 || fns.empty()) {
        if (terms.empty())
            throw std::runtime_error("GPEngine::grow: no terminals for type");
        std::uniform_int_distribution<size_t> d(0, terms.size() - 1);
        const Primitive* p = terms[d(rng_)];
        return std::make_unique<ExprNode>(p->name, p->returnType);
    }

    // Decide: terminal or function
    std::vector<const Primitive*> all;
    all.insert(all.end(), fns.begin(), fns.end());
    all.insert(all.end(), terms.begin(), terms.end());
    std::uniform_int_distribution<size_t> d(0, all.size() - 1);
    const Primitive* chosen = all[d(rng_)];

    auto node = std::make_unique<ExprNode>(chosen->name, chosen->returnType);
    for (GPType argT : chosen->argTypes) {
        node->children.push_back(grow(maxDepth - 1, argT));
    }
    return node;
}

std::unique_ptr<ExprNode> GPEngine::full(int depth, GPType targetType) {
    auto fns  = cfg_.stronglyTyped ? pset_.functionsOfType(targetType)
                                   : pset_.functionsOfType(GPType::Any);
    auto terms = cfg_.stronglyTyped ? pset_.terminalsOfType(targetType)
                                    : pset_.terminalsOfType(GPType::Any);

    if (fns.empty() && terms.empty())
        throw std::runtime_error("GPEngine::full: no primitives for type");

    if (depth == 0 || fns.empty()) {
        if (terms.empty())
            throw std::runtime_error("GPEngine::full: no terminals for type");
        std::uniform_int_distribution<size_t> d(0, terms.size() - 1);
        const Primitive* p = terms[d(rng_)];
        return std::make_unique<ExprNode>(p->name, p->returnType);
    }

    std::uniform_int_distribution<size_t> d(0, fns.size() - 1);
    const Primitive* chosen = fns[d(rng_)];

    auto node = std::make_unique<ExprNode>(chosen->name, chosen->returnType);
    for (GPType argT : chosen->argTypes) {
        node->children.push_back(full(depth - 1, argT));
    }
    return node;
}

std::vector<GPEngine::Ind> GPEngine::initPopulation(const FitnessFunc& f) {
    std::vector<Ind> pop;
    pop.reserve(cfg_.populationSize);

    // Ramped half-and-half: alternate grow / full across increasing depths
    for (int i = 0; i < cfg_.populationSize; ++i) {
        int depth = 2 + (i % (cfg_.initMaxDepth - 1));
        bool useGrow = (i % 2 == 0);
        Ind ind;
        ind.tree = useGrow ? grow(depth, cfg_.rootType)
                           : full(depth, cfg_.rootType);
        ind.fit  = f(*ind.tree);
        pop.push_back(std::move(ind));
    }
    return pop;
}

// ============================================================================
// Genetic operators
// ============================================================================

void GPEngine::collectNodes(ExprNode& node, std::vector<ExprNode*>& out) {
    out.push_back(&node);
    for (auto& ch : node.children)
        collectNodes(*ch, out);
}

std::unique_ptr<ExprNode> GPEngine::subtreeCrossover(const ExprNode& p1,
                                                      const ExprNode& p2) {
    // Clone both parents
    auto c1 = p1.clone();
    auto c2 = p2.clone();

    // Collect nodes from the children (we don't traverse with raw pointers into
    // the original trees — we work on the clones)
    std::vector<ExprNode*> nodes1, nodes2;
    collectNodes(*c1, nodes1);
    collectNodes(*c2, nodes2);

    std::uniform_int_distribution<size_t> d1(0, nodes1.size() - 1);
    std::uniform_int_distribution<size_t> d2(0, nodes2.size() - 1);

    ExprNode* pt1 = nodes1[d1(rng_)];
    ExprNode* pt2 = nodes2[d2(rng_)];

    // Swap subtrees by exchanging value, type, and children
    std::swap(pt1->value,    pt2->value);
    std::swap(pt1->type,     pt2->type);
    std::swap(pt1->children, pt2->children);

    // Enforce depth limit: if result is too deep, fall back to a terminal
    if (c1->depth() > cfg_.maxTreeDepth) {
        auto terms = pset_.terminalsOfType(GPType::Any);
        if (!terms.empty()) {
            std::uniform_int_distribution<size_t> td(0, terms.size() - 1);
            const Primitive* t = terms[td(rng_)];
            c1 = std::make_unique<ExprNode>(t->name, t->returnType);
        }
    }
    return c1;
}

void GPEngine::pointMutation(ExprNode& tree, int /*depth*/) {
    // Collect all nodes
    std::vector<ExprNode*> nodes;
    collectNodes(tree, nodes);
    if (nodes.empty()) return;

    std::uniform_int_distribution<size_t> pick(0, nodes.size() - 1);
    ExprNode* target = nodes[pick(rng_)];

    // Determine candidates with the same arity
    auto& functions = pset_.functions();
    auto& terminals = pset_.terminals();

    if (target->isLeaf()) {
        // Replace with a different terminal of the same type
        std::vector<const Primitive*> cands;
        for (const auto& p : terminals)
            if (!cfg_.stronglyTyped || p.returnType == target->type)
                cands.push_back(&p);
        if (cands.empty()) return;
        std::uniform_int_distribution<size_t> d(0, cands.size() - 1);
        const Primitive* chosen = cands[d(rng_)];
        target->value = chosen->name;
        target->type  = chosen->returnType;
    } else {
        // Replace with a function of the same arity (and compatible type)
        std::vector<const Primitive*> cands;
        for (const auto& p : functions)
            if ((int)p.argTypes.size() == target->arity() &&
                (!cfg_.stronglyTyped || p.returnType == target->type))
                cands.push_back(&p);
        if (cands.empty()) return;
        std::uniform_int_distribution<size_t> d(0, cands.size() - 1);
        const Primitive* chosen = cands[d(rng_)];
        target->value = chosen->name;
        target->type  = chosen->returnType;
        // Update child types if strongly typed
        if (cfg_.stronglyTyped) {
            for (int i = 0; i < (int)target->children.size(); ++i)
                target->children[i]->type = chosen->argTypes[i];
        }
    }
}

// ============================================================================
// Selection
// ============================================================================

const GPEngine::Ind& GPEngine::tournamentSelect(const std::vector<Ind>& pop, int k) {
    std::uniform_int_distribution<size_t> d(0, pop.size() - 1);
    size_t best = d(rng_);
    for (int i = 1; i < k; ++i) {
        size_t c = d(rng_);
        if (pop[c].fit > pop[best].fit) best = c;
    }
    return pop[best];
}

// ============================================================================
// Main run loop
// ============================================================================

GPResult GPEngine::run(const FitnessFunc& fitness) {
    auto pop = initPopulation(fitness);

    GPResult res;
    res.bestHistory.reserve(cfg_.generations);
    res.avgHistory.reserve(cfg_.generations);

    auto computeStats = [&](const std::vector<Ind>& P) {
        double sum = 0.0, best = -1e300;
        size_t bi = 0;
        for (size_t i = 0; i < P.size(); ++i) {
            sum += P[i].fit;
            if (P[i].fit > best) { best = P[i].fit; bi = i; }
        }
        if (res.bestTree == nullptr || best > res.bestFitness) {
            res.bestFitness = best;
            res.bestTree    = P[bi].tree->clone();
        }
        res.bestHistory.push_back(best);
        res.avgHistory.push_back(sum / P.size());
    };

    computeStats(pop);

    std::uniform_real_distribution<double> prob(0.0, 1.0);

    for (int gen = 0; gen < cfg_.generations; ++gen) {
        int elites = std::max(0, (int)std::round(cfg_.eliteRatio * pop.size()));

        std::vector<Ind> next;
        next.reserve(pop.size());

        // Elitism: keep top individuals
        if (elites > 0) {
            std::vector<size_t> idx(pop.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::nth_element(idx.begin(), idx.begin() + elites, idx.end(),
                [&](size_t a, size_t b){ return pop[a].fit > pop[b].fit; });
            for (int i = 0; i < elites; ++i) {
                Ind copy;
                copy.tree = pop[idx[i]].tree->clone();
                copy.fit  = pop[idx[i]].fit;
                next.push_back(std::move(copy));
            }
        }

        // Fill remainder
        while ((int)next.size() < cfg_.populationSize) {
            const Ind& parent1 = tournamentSelect(pop);
            Ind child;

            if (prob(rng_) < cfg_.crossoverRate) {
                const Ind& parent2 = tournamentSelect(pop);
                child.tree = subtreeCrossover(*parent1.tree, *parent2.tree);
            } else {
                child.tree = parent1.tree->clone();
            }

            if (prob(rng_) < cfg_.mutationRate)
                pointMutation(*child.tree);

            child.fit = fitness(*child.tree);
            next.push_back(std::move(child));
        }

        pop.swap(next);
        computeStats(pop);
    }

    return res;
}

} // namespace gp
} // namespace ga
