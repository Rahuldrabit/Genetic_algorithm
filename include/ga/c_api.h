#ifndef GA_C_API_H
#define GA_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ga_handle ga_handle;

typedef struct ga_config_c {
    int population_size;
    int generations;
    int dimension;
    double crossover_rate;
    double mutation_rate;
    double lower_bound;
    double upper_bound;
    double elite_ratio;
    unsigned int seed;
} ga_config_c;

typedef double (*ga_fitness_fn)(const double* genes, int length, void* user_data);

enum ga_status_code {
    GA_STATUS_OK = 0,
    GA_STATUS_INVALID_ARGUMENT = -1,
    GA_STATUS_RUNTIME_ERROR = -2
};

int ga_validate_config(const ga_config_c* cfg);
ga_handle* ga_create(const ga_config_c* cfg);
int ga_run(ga_handle* handle, ga_fitness_fn fitness, void* user_data);

double ga_best_fitness(const ga_handle* handle);
int ga_result_dimension(const ga_handle* handle);

// Copies best genes into out_genes and returns number of values written.
// Returns a negative ga_status_code on error.
int ga_best_genes(const ga_handle* handle, double* out_genes, int max_len);

int ga_history_length(const ga_handle* handle);

// Copies best-fitness history values and returns number of values written.
// Returns a negative ga_status_code on error.
int ga_best_history(const ga_handle* handle, double* out_values, int max_len);

// Copies average-fitness history values and returns number of values written.
// Returns a negative ga_status_code on error.
int ga_avg_history(const ga_handle* handle, double* out_values, int max_len);

const char* ga_last_error(void);
void ga_destroy(ga_handle* handle);

#ifdef __cplusplus
}
#endif

#endif // GA_C_API_H
