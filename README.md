# genetic-algorithm-lib

Genetic algorithm / evolutionary computation toolkit with a **C++17 core** and **Python bindings**.

Topics/tags: `genetic-algorithm`, `optimization`, `python`, `evolutionary-computation`

## What problem it solves

Use this package when you need a practical evolutionary optimizer for:

- black-box optimization (no gradients)
- parameter tuning / hyperparameter search
- feature selection and discrete search
- permutation / combinatorial problems (e.g., ordering)

The Python API is designed for “define a fitness function → configure → run → inspect results”.

## Installation

```bash
pip install genetic-algorithm-lib
```

Python: 3.8+

Import name (Python):

```python
import genetic_algorithm_lib as ga
```

## Example usage (Python)

```python
import genetic_algorithm_lib as ga

def sphere(x):
    # Fitness is maximized (higher is better)
    return 1000.0 / (1.0 + sum(xi * xi for xi in x))

cfg = ga.Config()
cfg.population_size = 50
cfg.generations = 100
cfg.dimension = 10
cfg.bounds = ga.Bounds(-5.12, 5.12)
cfg.seed = 42

engine = ga.GeneticAlgorithm(cfg)
result = engine.run(sphere)

print("Best fitness:", result.best_fitness)
print("Best genes:", result.best_genes)
print("History length:", len(result.best_history))
```

## Documentation

- Full guide (C++ + Python examples, operators, benchmarks, advanced features): [USER_GUIDE.md](USER_GUIDE.md)

## Features

- Multiple representations and operator families (crossover, mutation, selection)
- Single-objective GA engine + additional algorithms available in the core
- Benchmark suite (operator speed, function optimization, scalability)
- Plugin-style registries (create/register operators by name in Python)
- Optional parallel evaluation helpers

## Supported representations

The core supports several chromosome representations, including:

- Binary
- Real-valued (continuous)
- Integer
- Permutation

Python bindings expose the representation-aware operators; see the user guide for the exact surface.

## Running benchmarks (results + how to interpret)

Benchmarks are useful to compare operators, validate changes, and understand tradeoffs.

### From Python (installed package)

```python
import genetic_algorithm_lib as ga

bcfg = ga.BenchmarkConfig()
bcfg.warmup_iterations = 1
bcfg.benchmark_iterations = 10
bcfg.verbose = False

bench = ga.GABenchmark(bcfg)
bench.run_all_benchmarks()  # can take time depending on iterations

# Inspect structured results
operator_results = bench.operator_results()
function_results = bench.function_results()
scalability_results = bench.scalability_results()

# Export to CSV for plotting
bench.export_to_csv("ga_benchmark_results.csv")
```

### From source (CMake)

```bash
mkdir -p build
cmake -S . -B build
cmake --build build
./build/bin/ga-benchmark --help
```

Understanding output (high level):

- Operator benchmarks: time/throughput per operator + representation
- Function benchmarks: best/avg fitness, time, convergence history
- Scalability benchmarks: how runtime changes with population/dimension

## Architecture and efficiency

- The heavy work runs in native C++ (fast inner loops, minimal allocations).
- Python calls into the compiled extension module (`_core`) via pybind11.
- Your fitness function is a Python callback; benchmark/parallel helpers exist to reduce overhead where needed.

## Development output

For a basic GA run, you typically use:

- `result.best_genes`: best solution vector
- `result.best_fitness`: fitness of the best solution
- `result.best_history`: best fitness per generation
- `result.avg_history`: average fitness per generation

These arrays are easy to plot or export to CSV.

## From future versions (planned / under consideration)

- More high-level convenience APIs around advanced algorithms
- Broader representation/operator coverage in Python where still partial
- More packaging/CI hardening and additional test coverage

---

## Building with CMake (from source)

Prerequisites:

- CMake 3.16+
- A C++17 compiler (GCC/Clang/MSVC)

Build:

```bash
mkdir -p build
cmake -S . -B build
cmake --build build
```

### CMake targets

- `genetic_algorithm` (static library)
- `simple-ga-test` (interactive demo; output binary: `simple_ga_test`)
- `ga-benchmark` (benchmark executable)
- Convenience custom targets: `run`, `benchmark`, `clean-results`
- Sanity executables: `operators-sanity`, `nsga2-sanity`, `nsga3-sanity`, `c-api-sanity`, ...

Examples:

```bash
cmake --build build --target ga-benchmark
cmake --build build --target simple-ga-test
cmake --build build --target nsga3-sanity

# Or use convenience targets
cmake --build build --target run
cmake --build build --target benchmark
```

## Interactive mode

If you build from source, you can run the interactive demo:

```bash
./build/bin/simple_ga_test
```

It guides you through representation selection and compatible operator choices.

## Configuration

Python configuration uses `ga.Config` (snake_case fields):

```python
import genetic_algorithm_lib as ga

cfg = ga.Config()
cfg.population_size = 60
cfg.generations = 200
cfg.dimension = 10
cfg.bounds = ga.Bounds(-5.12, 5.12)
cfg.crossover_rate = 0.8
cfg.mutation_rate = 0.05
cfg.elite_ratio = 0.05
cfg.seed = 1
```

## Contributing

- Keep changes focused and add/update tests when applicable.
- Prefer small PRs: one feature/fix at a time.

## License

Apache-2.0

## Troubleshooting

- Build from source requires a working compiler toolchain + CMake.
- If you’re installing from PyPI, prefer upgrading pip (`python -m pip install -U pip`) to ensure you get the correct wheel for your platform.

## Common issues

- `ImportError` / missing compiled module: you may be on a platform without a prebuilt wheel; try installing from source (requires CMake + compiler).
- Compilation fails on Linux: install a compiler toolchain (e.g., `build-essential`) and CMake.
- Slow runs with expensive Python fitness: consider reducing Python overhead (vectorized fitness where possible) or using the provided parallel evaluation helpers.
