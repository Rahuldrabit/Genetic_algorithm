#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ga {
namespace plugin {

template <typename Base>
class Registry {
public:
    using Factory = std::function<std::unique_ptr<Base>()>;

    void registerFactory(const std::string& name, Factory factory) {
        if (!factory) {
            throw std::invalid_argument("Cannot register empty factory for: " + name);
        }
        factories_[name] = std::move(factory);
    }

    bool has(const std::string& name) const {
        return factories_.find(name) != factories_.end();
    }

    std::unique_ptr<Base> create(const std::string& name) const {
        auto it = factories_.find(name);
        if (it == factories_.end()) {
            throw std::out_of_range("Factory not found in registry: " + name);
        }
        return it->second();
    }

    std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(factories_.size());
        for (const auto& kv : factories_) {
            out.push_back(kv.first);
        }
        return out;
    }

private:
    std::unordered_map<std::string, Factory> factories_;
};

} // namespace plugin
} // namespace ga
