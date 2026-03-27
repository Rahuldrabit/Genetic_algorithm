#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ga/core/genome.hpp"

namespace ga {
namespace representations {

class NdArrayGenome : public ga::IGenome {
public:
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> data;

    NdArrayGenome() = default;
    NdArrayGenome(std::size_t r, std::size_t c)
        : rows(r), cols(c), data(r * c, 0.0) {}

    double& at(std::size_t r, std::size_t c) {
        if (r >= rows || c >= cols) {
            throw std::out_of_range("NdArrayGenome index out of range");
        }
        return data[r * cols + c];
    }

    const double& at(std::size_t r, std::size_t c) const {
        if (r >= rows || c >= cols) {
            throw std::out_of_range("NdArrayGenome index out of range");
        }
        return data[r * cols + c];
    }

    std::unique_ptr<ga::IGenome> clone() const override {
        return std::make_unique<NdArrayGenome>(*this);
    }

    std::string encodingName() const override {
        return "ndarray";
    }
};

} // namespace representations
} // namespace ga
