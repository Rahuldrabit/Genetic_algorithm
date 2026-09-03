#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "ga/evaluation/distributed_executor.hpp"

namespace {

bool approx(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) <= eps;
}

} // namespace

int main() {
    try {
        ga::evaluation::ProcessDistributedExecutor exec(
            [](const std::vector<double>& x) {
                double s = 0.0;
                for (double v : x) {
                    s += v * v;
                }
                return s;
            },
            3);

#if defined(__unix__) || defined(__APPLE__)
        const std::vector<std::vector<double>> batch = {
            {1.0, 2.0, 2.0},
            {3.0, 4.0},
            {5.0},
            {0.0, 6.0},
            {-2.0, 1.0, 2.0},
        };

        const auto out = exec.execute(batch);
        if (out.size() != batch.size()) {
            throw std::runtime_error("output size mismatch");
        }

        const std::vector<double> expected = {9.0, 25.0, 25.0, 36.0, 9.0};
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (!approx(out[i], expected[i])) {
                throw std::runtime_error("mismatched result at index " + std::to_string(i));
            }
        }

        const auto empty = exec.execute({});
        if (!empty.empty()) {
            throw std::runtime_error("expected empty result for empty input batch");
        }

        std::cout << "[PASS] Process distributed executor sanity checks\n";
        return 0;
#else
        bool unsupportedRejected = false;
        try {
            (void)exec.execute({{1.0}});
        } catch (const std::runtime_error&) {
            unsupportedRejected = true;
        }
        if (!unsupportedRejected) {
            throw std::runtime_error(
                "non-POSIX process executor did not report unsupported operation");
        }
        std::cout << "[PASS] Process executor reports unsupported platform\n";
        return 0;
#endif
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}
