#include "ga/c_api.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "ga/config.hpp"
#include "ga/genetic_algorithm.hpp"

namespace {

thread_local std::string g_last_error;

void set_error(const std::string& msg) {
    g_last_error = msg;
}

void clear_error() {
    g_last_error.clear();
}

int validate_config_raw(const ga_config_c* cfg, std::string* reason) {
    if (!cfg) {
        if (reason) *reason = "config is null";
        return GA_STATUS_INVALID_ARGUMENT;
    }
    if (cfg->population_size <= 0) {
        if (reason) *reason = "population_size must be > 0";
        return GA_STATUS_INVALID_ARGUMENT;
    }
    if (cfg->generations < 0) {
        if (reason) *reason = "generations must be >= 0";
        return GA_STATUS_INVALID_ARGUMENT;
    }
    if (cfg->dimension <= 0) {
        if (reason) *reason = "dimension must be > 0";
        return GA_STATUS_INVALID_ARGUMENT;
    }
    if (cfg->crossover_rate < 0.0 || cfg->crossover_rate > 1.0) {
        if (reason) *reason = "crossover_rate must be in [0,1]";
        return GA_STATUS_INVALID_ARGUMENT;
    }
    if (cfg->mutation_rate < 0.0 || cfg->mutation_rate > 1.0) {
        if (reason) *reason = "mutation_rate must be in [0,1]";
        return GA_STATUS_INVALID_ARGUMENT;
    }
    if (cfg->lower_bound >= cfg->upper_bound) {
        if (reason) *reason = "lower_bound must be < upper_bound";
        return GA_STATUS_INVALID_ARGUMENT;
    }
    if (cfg->elite_ratio < 0.0 || cfg->elite_ratio > 1.0) {
        if (reason) *reason = "elite_ratio must be in [0,1]";
        return GA_STATUS_INVALID_ARGUMENT;
    }
    return GA_STATUS_OK;
}

ga::Config to_cpp_config(const ga_config_c* cfg) {
    ga::Config out;
    out.populationSize = cfg->population_size;
    out.generations = cfg->generations;
    out.dimension = cfg->dimension;
    out.crossoverRate = cfg->crossover_rate;
    out.mutationRate = cfg->mutation_rate;
    out.bounds = {cfg->lower_bound, cfg->upper_bound};
    out.eliteRatio = cfg->elite_ratio;
    out.seed = cfg->seed;
    return out;
}

} // namespace

struct ga_handle {
    ga::Config config;
    std::unique_ptr<ga::GeneticAlgorithm> engine;
    ga::Result last_result;
    int has_result = 0;
};

extern "C" {

int ga_validate_config(const ga_config_c* cfg) {
    clear_error();
    std::string reason;
    const int status = validate_config_raw(cfg, &reason);
    if (status != GA_STATUS_OK) {
        set_error("ga_validate_config: " + reason);
    }
    return status;
}

ga_handle* ga_create(const ga_config_c* cfg) {
    clear_error();
    std::string reason;
    const int status = validate_config_raw(cfg, &reason);
    if (status != GA_STATUS_OK) {
        set_error("ga_create: " + reason);
        return nullptr;
    }

    try {
        std::unique_ptr<ga_handle> handle(new ga_handle());
        handle->config = to_cpp_config(cfg);
        handle->engine = std::make_unique<ga::GeneticAlgorithm>(handle->config);
        return handle.release();
    } catch (const std::exception& e) {
        set_error(std::string("ga_create failed: ") + e.what());
    } catch (...) {
        set_error("ga_create failed: unknown error");
    }
    return nullptr;
}

int ga_run(ga_handle* handle, ga_fitness_fn fitness, void* user_data) {
    clear_error();
    if (!handle || !handle->engine) {
        set_error("ga_run: handle is null or invalid");
        return GA_STATUS_INVALID_ARGUMENT;
    }
    if (!fitness) {
        set_error("ga_run: fitness callback is null");
        return GA_STATUS_INVALID_ARGUMENT;
    }

    try {
        auto wrapped = [fitness, user_data](const std::vector<double>& genes) {
            return fitness(genes.data(), static_cast<int>(genes.size()), user_data);
        };
        handle->last_result = handle->engine->run(wrapped);
        handle->has_result = 1;
        return GA_STATUS_OK;
    } catch (const std::exception& e) {
        set_error(std::string("ga_run failed: ") + e.what());
        return GA_STATUS_RUNTIME_ERROR;
    } catch (...) {
        set_error("ga_run failed: unknown error");
        return GA_STATUS_RUNTIME_ERROR;
    }
}

double ga_best_fitness(const ga_handle* handle) {
    clear_error();
    if (!handle || !handle->has_result) {
        set_error("ga_best_fitness: no available result");
        return std::numeric_limits<double>::quiet_NaN();
    }
    return handle->last_result.bestFitness;
}

int ga_result_dimension(const ga_handle* handle) {
    clear_error();
    if (!handle || !handle->has_result) {
        set_error("ga_result_dimension: no available result");
        return GA_STATUS_INVALID_ARGUMENT;
    }
    return static_cast<int>(handle->last_result.bestGenes.size());
}

int ga_best_genes(const ga_handle* handle, double* out_genes, int max_len) {
    clear_error();
    if (!handle || !handle->has_result) {
        set_error("ga_best_genes: no available result");
        return GA_STATUS_INVALID_ARGUMENT;
    }
    if (!out_genes || max_len <= 0) {
        set_error("ga_best_genes: output buffer is null or max_len <= 0");
        return GA_STATUS_INVALID_ARGUMENT;
    }

    const int copy_count = std::min(max_len, static_cast<int>(handle->last_result.bestGenes.size()));
    for (int i = 0; i < copy_count; ++i) {
        out_genes[i] = handle->last_result.bestGenes[static_cast<std::size_t>(i)];
    }
    return copy_count;
}

int ga_history_length(const ga_handle* handle) {
    clear_error();
    if (!handle || !handle->has_result) {
        set_error("ga_history_length: no available result");
        return GA_STATUS_INVALID_ARGUMENT;
    }
    return static_cast<int>(handle->last_result.bestHistory.size());
}

int ga_best_history(const ga_handle* handle, double* out_values, int max_len) {
    clear_error();
    if (!handle || !handle->has_result) {
        set_error("ga_best_history: no available result");
        return GA_STATUS_INVALID_ARGUMENT;
    }
    if (!out_values || max_len <= 0) {
        set_error("ga_best_history: output buffer is null or max_len <= 0");
        return GA_STATUS_INVALID_ARGUMENT;
    }

    const int copy_count = std::min(max_len, static_cast<int>(handle->last_result.bestHistory.size()));
    for (int i = 0; i < copy_count; ++i) {
        out_values[i] = handle->last_result.bestHistory[static_cast<std::size_t>(i)];
    }
    return copy_count;
}

int ga_avg_history(const ga_handle* handle, double* out_values, int max_len) {
    clear_error();
    if (!handle || !handle->has_result) {
        set_error("ga_avg_history: no available result");
        return GA_STATUS_INVALID_ARGUMENT;
    }
    if (!out_values || max_len <= 0) {
        set_error("ga_avg_history: output buffer is null or max_len <= 0");
        return GA_STATUS_INVALID_ARGUMENT;
    }

    const int copy_count = std::min(max_len, static_cast<int>(handle->last_result.avgHistory.size()));
    for (int i = 0; i < copy_count; ++i) {
        out_values[i] = handle->last_result.avgHistory[static_cast<std::size_t>(i)];
    }
    return copy_count;
}

const char* ga_last_error(void) {
    return g_last_error.c_str();
}

void ga_destroy(ga_handle* handle) {
    delete handle;
}

} // extern "C"
