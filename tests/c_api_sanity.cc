#include <cmath>
#include <iostream>
#include <vector>

#include "ga/c_api.h"

static double sphere_fitness(const double* genes, int length, void* /*user_data*/) {
    double sum = 0.0;
    for (int i = 0; i < length; ++i) {
        sum += genes[i] * genes[i];
    }
    return 1000.0 / (1.0 + sum);
}

int main() {
    ga_config_c invalid_cfg{};
    invalid_cfg.population_size = 0;
    invalid_cfg.generations = 10;
    invalid_cfg.dimension = 4;
    invalid_cfg.crossover_rate = 0.8;
    invalid_cfg.mutation_rate = 0.1;
    invalid_cfg.lower_bound = -5.0;
    invalid_cfg.upper_bound = 5.0;
    invalid_cfg.elite_ratio = 0.05;
    invalid_cfg.seed = 123;

    if (ga_validate_config(&invalid_cfg) != GA_STATUS_INVALID_ARGUMENT) {
        std::cerr << "[FAIL] invalid config should fail validation\n";
        return 1;
    }

    ga_config_c cfg{};
    cfg.population_size = 40;
    cfg.generations = 30;
    cfg.dimension = 8;
    cfg.crossover_rate = 0.8;
    cfg.mutation_rate = 0.1;
    cfg.lower_bound = -5.0;
    cfg.upper_bound = 5.0;
    cfg.elite_ratio = 0.05;
    cfg.seed = 321;

    if (ga_validate_config(&cfg) != GA_STATUS_OK) {
        std::cerr << "[FAIL] valid config rejected: " << ga_last_error() << "\n";
        return 1;
    }

    ga_handle* handle = ga_create(&cfg);
    if (!handle) {
        std::cerr << "[FAIL] ga_create failed: " << ga_last_error() << "\n";
        return 1;
    }

    if (ga_run(handle, sphere_fitness, nullptr) != GA_STATUS_OK) {
        std::cerr << "[FAIL] ga_run failed: " << ga_last_error() << "\n";
        ga_destroy(handle);
        return 1;
    }

    const int dim = ga_result_dimension(handle);
    if (dim <= 0) {
        std::cerr << "[FAIL] invalid result dimension\n";
        ga_destroy(handle);
        return 1;
    }

    std::vector<double> genes(static_cast<std::size_t>(dim), 0.0);
    const int copied_genes = ga_best_genes(handle, genes.data(), dim);
    if (copied_genes != dim) {
        std::cerr << "[FAIL] best genes copy mismatch\n";
        ga_destroy(handle);
        return 1;
    }

    const double best = ga_best_fitness(handle);
    if (!std::isfinite(best)) {
        std::cerr << "[FAIL] best fitness is not finite\n";
        ga_destroy(handle);
        return 1;
    }

    const int hist_len = ga_history_length(handle);
    if (hist_len <= 0) {
        std::cerr << "[FAIL] history length should be positive\n";
        ga_destroy(handle);
        return 1;
    }

    std::vector<double> best_hist(static_cast<std::size_t>(hist_len), 0.0);
    std::vector<double> avg_hist(static_cast<std::size_t>(hist_len), 0.0);

    const int copied_best_hist = ga_best_history(handle, best_hist.data(), hist_len);
    const int copied_avg_hist = ga_avg_history(handle, avg_hist.data(), hist_len);
    if (copied_best_hist != hist_len || copied_avg_hist != hist_len) {
        std::cerr << "[FAIL] history copy mismatch\n";
        ga_destroy(handle);
        return 1;
    }

    ga_destroy(handle);
    std::cout << "[PASS] C API sanity checks\n";
    return 0;
}
