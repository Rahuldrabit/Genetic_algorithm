#pragma once

#include <memory>
#include <set>
#include <string>

#include "ga/core/genome.hpp"

namespace ga {
namespace representations {

class SetGenome : public ga::IGenome {
public:
    std::set<int> values;

    SetGenome() = default;
    explicit SetGenome(std::set<int> v) : values(std::move(v)) {}

    std::unique_ptr<ga::IGenome> clone() const override {
        return std::make_unique<SetGenome>(*this);
    }

    std::string encodingName() const override {
        return "set<int>";
    }
};

} // namespace representations
} // namespace ga
