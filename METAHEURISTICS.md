# Metaheuristics suite

The C++17 core provides interoperable population-based optimizers for continuous,
binary, graph, and clustering problems. Include the complete surface with:

```cpp
#include <ga/metaheuristics.hpp>
```

All continuous optimizers maximize `ga::Fitness`. Convert a minimization
objective to fitness, for example `1.0 / (1.0 + cost)` or `-cost`. Graph ACO
directly minimizes edge cost.

## Implemented algorithms

| Family | Implemented variants |
|---|---|
| PSO | global-best, ring local-best, constriction, bare-bones, fully informed, quantum-behaved, binary |
| Graph ACO | Ant System, Elitist Ant System, Rank-Based Ant System, Ant Colony System, MAX-MIN Ant System |
| Continuous ACO | ACOR (archive-based continuous ant colony optimization) |
| GSA | continuous gravitational search with decreasing K-best agents |
| Fuzzy | fuzzy C-means clustering and a reusable zero-order Sugeno adaptive controller |
| Hybrid | sequential heterogeneous pipeline with solution transfer, including a GA adapter |

“All ACO/PSO types” is not a finite category in the literature. This release
implements the major distinct families above; new variants can implement
`IContinuousOptimizer` and immediately participate in the hybrid pipeline.

## User-controlled design

The library does not select an algorithm, construct a hybrid, reorder stages,
or enable adaptation automatically. Each optimizer can run independently. A
hybrid exists only when the caller creates a `MetaheuristicPipeline` and calls
`add(...)` in the exact desired order; the pipeline preserves that order and
only transfers solutions between those explicitly selected stages.

All algorithm hyperparameters are exposed through public configuration structs.
Their defaults are conveniences, not an automatic tuning policy. An adaptive
controller is also opt-in: leaving `controller` unset keeps the configured
algorithm parameters fixed. `FuzzyControllerConfig` exposes its membership
thresholds, improvement scale, and every Sugeno rule consequent.

Robustness here means validated configurations and controller outputs, bounded
solutions, rejection of non-finite fitness values, deterministic seeded runs,
exact evaluation accounting, and cross-platform tests. It does not mean hidden
algorithm selection or hidden hyperparameter tuning.

## Shared continuous interface

```cpp
class IContinuousOptimizer {
public:
    virtual std::string name() const = 0;
    virtual ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const SeedPopulation& seeds = {}) = 0;
};
```

`SearchConfig` controls population size, iteration budget, dimension, uniform
bounds, deterministic seed, and objective-evaluation threads. Results report the
best solution, best/average histories, completed iterations, and exact objective
evaluation count.

## User-selected hybrid with optional fuzzy control

```cpp
ga::fuzzy::FuzzyControllerConfig fuzzyCfg;
fuzzyCfg.improvementScale = 20.0;
fuzzyCfg.lowDiversityStagnant = {1.55, 0.80, 1.30, 1.45};
auto fuzzy = std::make_shared<ga::fuzzy::FuzzyAdaptiveController>(fuzzyCfg);

ga::Config gaCfg;
gaCfg.dimension = 10;
gaCfg.bounds = {-5.0, 5.0};

ga::pso::PsoConfig psoCfg;
psoCfg.search.dimension = 10;
psoCfg.search.bounds = {-5.0, 5.0};
psoCfg.variant = ga::pso::PsoVariant::Constriction;
psoCfg.controller = fuzzy; // Optional: omit for fixed PSO parameters.

ga::aco::AcorConfig acorCfg;
acorCfg.search.dimension = 10;
acorCfg.search.bounds = {-5.0, 5.0};
acorCfg.controller = fuzzy; // Optional.

ga::hybrid::MetaheuristicPipeline pipeline;
pipeline.add(std::make_unique<ga::metaheuristics::GeneticAlgorithmAdapter>(gaCfg))
        .add(std::make_unique<ga::pso::ParticleSwarmOptimizer>(psoCfg))
        .add(std::make_unique<ga::aco::ContinuousAntColonyOptimizer>(acorCfg));

auto result = pipeline.optimize(fitness);
```

The caller above explicitly chooses GA, then PSO, then ACOR. Removing, replacing,
or reordering an `add(...)` call changes the hybrid accordingly. The optional
controller observes normalized population diversity, recent improvement,
and stagnation. It returns dimensionless exploration, exploitation, evaporation,
and randomization multipliers. Each algorithm maps those signals to its native
parameters. A single shared controller can therefore adapt every stage without
coupling the algorithms to fuzzy-logic implementation details.

## Graph ACO example

```cpp
ga::aco::DenseGraph graph({
    {0, 2, 9, 10},
    {2, 0, 6, 4},
    {9, 6, 0, 8},
    {10, 4, 8, 0},
});

ga::aco::AntColonyConfig cfg;
cfg.variant = ga::aco::AntColonyVariant::MaxMinAntSystem;
cfg.controller = std::make_shared<ga::fuzzy::FuzzyAdaptiveController>();
auto tour = ga::aco::AntColonyOptimizer(cfg).solve(graph);
```

`DenseGraph` supports symmetric and directed complete graphs. Candidate lists,
precomputed heuristic values, contiguous pheromone storage, and allocation reuse
keep the tour-construction inner loop compact.

## Fuzzy C-means example

```cpp
ga::fuzzy::FuzzyCMeansConfig cfg;
cfg.clusters = 3;
cfg.fuzziness = 2.0;
auto clusters = ga::fuzzy::FuzzyCMeans(cfg).fit(data);
auto hardLabels = clusters.labels();
```

The result retains the soft membership matrix, centers, objective history,
convergence flag, and iteration count.

## Performance notes

- Fitness batches can be evaluated concurrently with `SearchConfig::threads`.
  The fitness callback must be safe for concurrent calls when `threads > 1`.
- Expensive buffers and graph heuristics are allocated or computed outside hot loops.
- The existing GA now evaluates each mutated offspring exactly once and caches its
  bound vectors, removing redundant objective calls and per-offspring allocations.
- Fixed seeds make algorithm comparisons reproducible.

Run the deterministic tests and release benchmark with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/tests/metaheuristics-sanity
./build/bin/metaheuristics-benchmark
```

## Python API

The same algorithms, configuration fields, optional fuzzy controller, seeded
solutions, results, and caller-defined pipeline order are exposed by the
`genetic_algorithm_lib` Python package. See
[`python/metaheuristics_example.py`](python/metaheuristics_example.py) for a
complete example and [`python/README.md`](python/README.md) for the API guide.
