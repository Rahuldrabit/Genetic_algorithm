#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "ga/core/genome.hpp"

namespace ga {
namespace representations {

class MapGenome : public ga::IGenome {
public:
    std::unordered_map<std::string, double> values;

    MapGenome() = default;
    explicit MapGenome(std::unordered_map<std::string, double> v) : values(std::move(v)) {}

    std::unique_ptr<ga::IGenome> clone() const override {
        return std::make_unique<MapGenome>(*this);
    }

    std::string encodingName() const override {
        return "map<string,double>";
    }
};

} // namespace representations
} // namespace ga
