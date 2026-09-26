#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ga/fuzzy/fuzzy_types.hpp"
#include "ga/fuzzy/fuzzy_rule.hpp"
#include "ga/fuzzy/linguistic_variable.hpp"
#include "ga/fuzzy/defuzzification.hpp"
#include "ga/fuzzy/t_norms.hpp"
#include "ga/fuzzy/s_norms.hpp"

namespace ga {
namespace fuzzy {

enum class ImplicationMethod { Minimum, AlgebraicProduct };

class MamdaniSystem {
public:
    MamdaniSystem() = default;

    void addInput(std::string name, double minVal, double maxVal) {
        inputs_.emplace(name, LinguisticVariable(name, Interval{minVal, maxVal}));
    }
    void addInputTerm(const std::string& varName, std::string termName, std::shared_ptr<IMembershipFunction> mf) {
        inputs_.at(varName).addTerm(std::move(termName), std::move(mf));
    }
    void addOutput(std::string name, double minVal, double maxVal) {
        outputs_.emplace(name, LinguisticVariable(name, Interval{minVal, maxVal}));
    }
    void addOutputTerm(const std::string& varName, std::string termName, std::shared_ptr<IMembershipFunction> mf) {
        outputs_.at(varName).addTerm(std::move(termName), std::move(mf));
    }
    void addRule(MamdaniRule rule) { rules_.push_back(std::move(rule)); }

    void addRule(const std::string& ruleString) {
        // Simple robust parser: "IF x IS A AND y IS B THEN z IS C"
        std::istringstream iss(ruleString);
        std::string token;
        std::vector<FuzzyProposition> antes;
        std::vector<MamdaniConsequent> cons;
        RuleConnector conn = RuleConnector::And;
        bool inThen = false;

        while (iss >> token) {
            std::string upper = token;
            for (char& c : upper) c = static_cast<char>(std::toupper(c));
            if (upper == "IF") continue;
            if (upper == "THEN") { inThen = true; continue; }
            if (upper == "AND") { conn = RuleConnector::And; continue; }
            if (upper == "OR") { conn = RuleConnector::Or; continue; }

            if (!inThen) {
                std::string varName = token;
                std::string isToken;
                if (iss >> isToken) {
                    std::string termName;
                    bool negated = false;
                    if (iss >> termName) {
                        std::string tUpper = termName;
                        for (char& c : tUpper) c = static_cast<char>(std::toupper(c));
                        if (tUpper == "NOT") {
                            negated = true;
                            iss >> termName;
                        }
                        antes.push_back({varName, termName, negated});
                    }
                }
            } else {
                std::string varName = token;
                std::string isToken, termName;
                if (iss >> isToken && iss >> termName) {
                    cons.push_back({varName, termName});
                }
            }
        }
        if (!antes.empty() && !cons.empty()) {
            rules_.emplace_back(std::move(antes), std::move(cons), conn);
        }
    }

    void setTNorm(TNormType tnorm) noexcept { tnorm_ = tnorm; }
    void setSNorm(SNormType snorm) noexcept { snorm_ = snorm; }
    void setImplication(ImplicationMethod imp) noexcept { implication_ = imp; }
    void setDefuzzification(DefuzzMethod method) noexcept { defuzzMethod_ = method; }
    void setResolution(std::size_t res) noexcept { resolution_ = res; }

    std::map<std::string, double> evaluate(const std::map<std::string, double>& inputs) const {
        std::map<std::string, double> results;
        for (const auto& [outName, outVar] : outputs_) {
            results[outName] = evaluateSingle(outName, inputs, outVar.domain().center());
        }
        return results;
    }

    double evaluateSingle(const std::string& outputVar, const std::map<std::string, double>& inputs, double fallback = 0.0) const {
        auto itOut = outputs_.find(outputVar);
        if (itOut == outputs_.end()) return fallback;
        const auto& outVar = itOut->second;

        // Collect rule firing strengths for this output variable
        struct ActiveTerm { double firing; std::string term; };
        std::vector<ActiveTerm> active;

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
            if (strength <= 1e-9) continue;

            for (const auto& con : rule.consequents()) {
                if (con.variable == outputVar) {
                    active.push_back({strength, con.term});
                }
            }
        }

        if (active.empty()) return fallback;

        auto curve = [&](double y) -> double {
            double agg = 0.0;
            for (const auto& at : active) {
                const double baseMu = outVar.evaluateTerm(at.term, y);
                const double clipped = (implication_ == ImplicationMethod::Minimum)
                    ? std::min(at.firing, baseMu)
                    : (at.firing * baseMu);
                agg = applySNorm(snorm_, agg, clipped);
            }
            return agg;
        };

        return defuzzify(curve, outVar.domain(), defuzzMethod_, resolution_, fallback);
    }

    const std::map<std::string, LinguisticVariable>& inputs() const noexcept { return inputs_; }
    const std::map<std::string, LinguisticVariable>& outputs() const noexcept { return outputs_; }
    const std::vector<MamdaniRule>& rules() const noexcept { return rules_; }

private:
    std::map<std::string, LinguisticVariable> inputs_;
    std::map<std::string, LinguisticVariable> outputs_;
    std::vector<MamdaniRule> rules_;
    TNormType tnorm_ = TNormType::Minimum;
    SNormType snorm_ = SNormType::Maximum;
    ImplicationMethod implication_ = ImplicationMethod::Minimum;
    DefuzzMethod defuzzMethod_ = DefuzzMethod::Centroid;
    std::size_t resolution_ = 200;
};

} // namespace fuzzy
} // namespace ga
