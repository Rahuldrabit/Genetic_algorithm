#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "ga/fuzzy/fuzzy_types.hpp"
#include "ga/fuzzy/membership_functions.hpp"

namespace ga {
namespace fuzzy {

struct LinguisticTerm {
    std::string name;
    std::shared_ptr<IMembershipFunction> membership;
};

class LinguisticVariable {
public:
    LinguisticVariable(std::string name, Interval domain)
        : name_(std::move(name)), domain_(domain) {}

    const std::string& name() const noexcept { return name_; }
    const Interval& domain() const noexcept { return domain_; }

    void addTerm(std::string termName, std::shared_ptr<IMembershipFunction> mf) {
        if (!mf) throw std::invalid_argument("LinguisticVariable::addTerm: membership function cannot be null");
        terms_.push_back({std::move(termName), std::move(mf)});
    }

    bool hasTerm(const std::string& termName) const {
        for (const auto& t : terms_) {
            if (t.name == termName) return true;
        }
        return false;
    }

    const IMembershipFunction& term(const std::string& termName) const {
        for (const auto& t : terms_) {
            if (t.name == termName && t.membership) return *t.membership;
        }
        throw std::out_of_range("LinguisticVariable: term not found: " + termName);
    }

    std::map<std::string, double> fuzzify(double x) const {
        std::map<std::string, double> res;
        for (const auto& t : terms_) {
            const double val = t.membership ? std::clamp(t.membership->evaluate(x), 0.0, 1.0) : 0.0;
            res[t.name] = val;
        }
        return res;
    }

    double evaluateTerm(const std::string& termName, double x) const {
        return std::clamp(term(termName).evaluate(x), 0.0, 1.0);
    }

    const std::vector<LinguisticTerm>& terms() const noexcept { return terms_; }

private:
    std::string name_;
    Interval domain_;
    std::vector<LinguisticTerm> terms_;
};

} // namespace fuzzy
} // namespace ga
