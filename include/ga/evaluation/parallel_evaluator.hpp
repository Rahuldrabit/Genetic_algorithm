#pragma once

#include <algorithm>
#include <future>
#include <thread>
#include <utility>
#include <vector>

namespace ga {
namespace evaluation {

template <typename T, typename R>
class IEvaluator {
public:
    virtual ~IEvaluator() = default;
    virtual std::vector<R> evaluate(const std::vector<T>& inputs) const = 0;
};

template <typename T, typename R, typename Fn>
class SerialEvaluator : public IEvaluator<T, R> {
public:
    explicit SerialEvaluator(Fn fn) : fn_(std::move(fn)) {}

    std::vector<R> evaluate(const std::vector<T>& inputs) const override {
        std::vector<R> out;
        out.reserve(inputs.size());
        for (const auto& in : inputs) {
            out.push_back(fn_(in));
        }
        return out;
    }

private:
    Fn fn_;
};

template <typename T, typename R, typename Fn>
class ParallelEvaluator : public IEvaluator<T, R> {
public:
    explicit ParallelEvaluator(Fn fn, std::size_t threads = std::thread::hardware_concurrency())
        : fn_(std::move(fn)), threads_(std::max<std::size_t>(1, threads)) {}

    std::vector<R> evaluate(const std::vector<T>& inputs) const override {
        std::vector<R> out(inputs.size());
        if (inputs.empty()) {
            return out;
        }

        const std::size_t batch = (inputs.size() + threads_ - 1) / threads_;
        std::vector<std::future<void>> tasks;

        for (std::size_t t = 0; t < threads_; ++t) {
            const std::size_t begin = t * batch;
            if (begin >= inputs.size()) {
                break;
            }
            const std::size_t end = std::min(inputs.size(), begin + batch);
            tasks.push_back(std::async(std::launch::async, [&, begin, end]() {
                for (std::size_t i = begin; i < end; ++i) {
                    out[i] = fn_(inputs[i]);
                }
            }));
        }

        for (auto& task : tasks) {
            task.get();
        }
        return out;
    }

private:
    Fn fn_;
    std::size_t threads_;
};

} // namespace evaluation
} // namespace ga
