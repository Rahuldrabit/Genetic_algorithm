#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ga/fuzzy/fuzzy_types.hpp"
#include "ga/fuzzy/fuzzy_rule.hpp"
#include "ga/fuzzy/linguistic_variable.hpp"
#include "ga/fuzzy/t_norms.hpp"
#include "ga/fuzzy/s_norms.hpp"

namespace ga {
namespace fuzzy {

class SugenoSystem {
public:
    SugenoSystem() = default;

    void addInput(std::string name, double minVal, double maxVal) {
        inputs_.emplace(name, LinguisticVariable(name, Interval{minVal, maxVal}));
    }
    void addInputTerm(const std::string& varName, std::string termName, std::shared_ptr<IMembershipFunction> mf) {
        inputs_.at(varName).addTerm(std::move(termName), std::move(mf));
    }
    void addOutput(std::string name, double fallback = 0.0) {
        outputs_[name] = fallback;
    }
    void addRule(SugenoRule rule) { rules_.push_back(std::move(rule)); }

    void setTNorm(TNormType tnorm) noexcept { tnorm_ = tnorm; }
    void setSNorm(SNormType snorm) noexcept { snorm_ = snorm; }

    std::map<std::string, double> evaluate(const std::map<std::string, double>& inputs) const {
        std::map<std::string, double> res;
        for (const auto& [name, fallback] : outputs_) {
            res[name] = evaluateSingle(name, inputs, fallback);
        }
        return res;
    }

    double evaluateSingle(const std::string& outputVar, const std::map<std::string, double>& inputs, double fallback = 0.0) const {
        double num = 0.0, denom = 0.0;
        for (const auto& rule : rules_) {
            if (rule.antecedents().empty()) continue;
            double strength = (rule.connector() == RuleConnector::And) ? 1.0 : 0.0;
            bool first = true;

            for (const auto& ante : rule.antecedents()) {
                auto itIn = inputs_.find(ante.variable);
                if (itIn == inputs_.end()) continue;
                auto itVal = inputs.find(ante.variable);
                const double x = (itVal != inputs.end()) ? itVal->second : itIn->second.domain().center();
                double mu = itIn->second.evaluateTerm(ante.term, x);
                if (ante.negated) mu = 1.0 - mu;

                if (first) {
                    strength = mu;
                    first = false;
                } else if (rule.connector() == RuleConnector::And) {
                    strength = applyTNorm(tnorm_, strength, mu);
                } else {
                    strength = applySNorm(snorm_, strength, mu);
                }
            }
            strength *= rule.weight();
            if (strength <= 1e-12) continue;

            for (const auto& con : rule.consequents()) {
                if (con.variable == outputVar) {
                    double val = con.constant;
                    for (const auto& [coefVar, coef] : con.coefficients) {
                        auto itX = inputs.find(coefVar);
                        const double x = (itX != inputs.end()) ? itX->second : 0.0;
                        val += coef * x;
                    }
                    num += strength * val;
                    denom += strength;
                }
            }
        }
        return (denom > 1e-12) ? (num / denom) : fallback;
    }

    const std::map<std::string, LinguisticVariable>& inputs() const noexcept { return inputs_; }
    const std::vector<SugenoRule>& rules() const noexcept { return rules_; }

private:
    std::map<std::string, LinguisticVariable> inputs_;
    std::map<std::string, double> outputs_;
    std::vector<SugenoRule> rules_;
    TNormType tnorm_ = TNormType::AlgebraicProduct;
    SNormType snorm_ = SNormType::Maximum;
};

} // namespace fuzzy
} // namespace ga
