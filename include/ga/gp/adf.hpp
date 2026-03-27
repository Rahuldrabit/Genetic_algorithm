#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "ga/gp/node.hpp"

namespace ga {
namespace gp {

class ADFPool {
public:
    void put(const std::string& name, const Node& root) {
        functions_[name] = root.clone();
    }

    bool has(const std::string& name) const {
        return functions_.find(name) != functions_.end();
    }

    const Node& get(const std::string& name) const {
        auto it = functions_.find(name);
        if (it == functions_.end() || !it->second) {
            throw std::out_of_range("ADF function not found: " + name);
        }
        return *(it->second);
    }

    std::size_t size() const {
        return functions_.size();
    }

private:
    std::unordered_map<std::string, std::unique_ptr<Node>> functions_;
};

} // namespace gp
} // namespace ga
