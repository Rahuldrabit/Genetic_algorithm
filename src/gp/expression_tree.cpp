#include "ga/gp/expression_tree.hpp"

#include <stdexcept>
#include <cmath>

namespace ga {
namespace gp {

double evaluate(const ExprNode& node,
                const PrimitiveSet& pset,
                const std::vector<double>& vars)
{
    // Try function primitives
    for (const auto& p : pset.functions()) {
        if (p.name == node.value && (int)p.argTypes.size() == node.arity()) {
            std::vector<double> childVals;
            childVals.reserve(node.children.size());
            for (const auto& ch : node.children)
                childVals.push_back(evaluate(*ch, pset, vars));
            return p.eval(childVals);
        }
    }

    // Try terminal primitives
    for (const auto& p : pset.terminals()) {
        if (p.name == node.value && node.isLeaf())
            return p.eval(vars);
    }

    // Numeric literal (ephemeral constant stored as a string)
    try {
        return std::stod(node.value);
    } catch (const std::exception&) {
        throw std::runtime_error("GP evaluate: unknown primitive '" + node.value + "'");
    }
}

} // namespace gp
} // namespace ga
