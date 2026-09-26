#pragma once

#include <string>
#include <vector>

namespace ga {
namespace fuzzy {

enum class RuleConnector {
    And,
    Or
};

struct FuzzyProposition {
    std::string variable;
    std::string term;
    bool negated = false;
};

struct MamdaniConsequent {
    std::string variable;
    std::string term;
};

struct SugenoConsequent {
    std::string variable;
    // Linear function: f(x) = c0 + sum(c_i * x_i)
    double constant = 0.0;
    std::vector<std::pair<std::string, double>> coefficients;
};

class MamdaniRule {
public:
    MamdaniRule(std::vector<FuzzyProposition> antecedents,
                std::vector<MamdaniConsequent> consequents,
                RuleConnector connector = RuleConnector::And,
                double weight = 1.0)
        : antecedents_(std::move(antecedents)),
          consequents_(std::move(consequents)),
          connector_(connector),
          weight_(weight) {}

    const std::vector<FuzzyProposition>& antecedents() const noexcept { return antecedents_; }
    const std::vector<MamdaniConsequent>& consequents() const noexcept { return consequents_; }
    RuleConnector connector() const noexcept { return connector_; }
    double weight() const noexcept { return weight_; }

    void setWeight(double w) noexcept { weight_ = w; }

private:
    std::vector<FuzzyProposition> antecedents_;
    std::vector<MamdaniConsequent> consequents_;
    RuleConnector connector_ = RuleConnector::And;
    double weight_ = 1.0;
};

class SugenoRule {
public:
    SugenoRule(std::vector<FuzzyProposition> antecedents,
               std::vector<SugenoConsequent> consequents,
               RuleConnector connector = RuleConnector::And,
               double weight = 1.0)
        : antecedents_(std::move(antecedents)),
          consequents_(std::move(consequents)),
          connector_(connector),
          weight_(weight) {}

    const std::vector<FuzzyProposition>& antecedents() const noexcept { return antecedents_; }
    const std::vector<SugenoConsequent>& consequents() const noexcept { return consequents_; }
    RuleConnector connector() const noexcept { return connector_; }
    double weight() const noexcept { return weight_; }

private:
    std::vector<FuzzyProposition> antecedents_;
    std::vector<SugenoConsequent> consequents_;
    RuleConnector connector_ = RuleConnector::And;
    double weight_ = 1.0;
};

} // namespace fuzzy
} // namespace ga
