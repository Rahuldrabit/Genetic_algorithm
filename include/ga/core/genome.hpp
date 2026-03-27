#pragma once

#include <memory>
#include <string>

namespace ga {

class IGenome {
public:
    virtual ~IGenome() = default;
    virtual std::unique_ptr<IGenome> clone() const = 0;
    virtual std::string encodingName() const = 0;
};

} // namespace ga
