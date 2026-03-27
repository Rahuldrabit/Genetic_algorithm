#include "ga/individual.hpp"
#include "ga/gp/expression_tree.hpp"

namespace ga {

double TreeIndividual::fitness() const {
    if (!root_) return -1e300;
    return fn_(*root_);
}

std::unique_ptr<Individual> TreeIndividual::clone() const {
    return std::make_unique<TreeIndividual>(
        root_ ? root_->clone() : nullptr,
        fn_);
}

std::string TreeIndividual::toString() const {
    return "TreeIndividual" + (root_ ? root_->toSExpr() : "[]");
}

} // namespace ga
