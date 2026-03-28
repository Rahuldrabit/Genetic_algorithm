# Genetic Algorithm Framework — User Guide

This guide documents **every implemented feature** with both **C++** and **Python** usage examples.
For features not yet exposed in the Python bindings, an explicit note is included.

> **Python binding symbols verified from `python/ga_bindings.cpp`.**
> All C++ examples assume `#include <ga/genetic_algorithm.hpp>` and the framework
> is built with CMake (see [README.md](README.md) for build instructions).

---

## Table of Contents — Feature Index

| # | Feature | C++ | Python |
|---|---------|-----|--------|
| 1 | [Quick Start](#1-quick-start) | ✅ | ✅ |
| 2 | [Configuration (`Config`, `Bounds`)](#2-configuration) | ✅ | ✅ |
| 3 | [Chromosome Representations](#3-chromosome-representations) | ✅ | ✅ (all genome types) |
| 4 | [Crossover Operators](#4-crossover-operators) | ✅ | ⚠️ 2 factory operators exposed |
| 5 | [Mutation Operators](#5-mutation-operators) | ✅ | ⚠️ 2 factory operators exposed |
| 6 | [Selection Operators](#6-selection-operators) | ✅ | ⚠️ helper functions exposed |
| 7 | [Core GA Run and Results](#7-core-ga-run-and-results) | ✅ | ✅ |
| 8 | [High-Level Optimizer API](#8-high-level-optimizer-api) | ✅ | ✅ |
| 9 | [Multi-Objective: NSGA-II](#9-multi-objective-nsga-ii) | ✅ | ✅ (objective-space utils) |
| 10 | [Multi-Objective: NSGA-III](#10-multi-objective-nsga-iii) | ✅ | ✅ |
| 11 | [Multi-Objective: SPEA2](#11-multi-objective-spea2) | ✅ | ✅ (objective-space utils) |
| 12 | [Multi-Objective: MO-CMA-ES](#12-multi-objective-mo-cma-es) | ✅ | ✅ |
| 13 | [Evolution Strategies (ES)](#13-evolution-strategies-es) | ✅ | ✅ |
| 14 | [CMA-ES](#14-cma-es) | ✅ | ✅ |
| 15 | [Genetic Programming (GP)](#15-genetic-programming-gp) | ✅ | ✅ |
| 16 | [Adaptive Operators](#16-adaptive-operators) | ✅ | ✅ |
| 17 | [Hybrid Optimization](#17-hybrid-optimization) | ✅ | ✅ |
| 18 | [Constraint Handling](#18-constraint-handling) | ✅ | ✅ |
| 19 | [Parallel and Distributed Evaluation](#19-parallel-and-distributed-evaluation) | ✅ | ✅ (`ParallelEvaluator`, `LocalDistributedExecutor`, `Optimizer.with_threads`) |
| 20 | [Co-Evolution](#20-co-evolution) | ✅ | ✅ |
| 21 | [Checkpointing](#21-checkpointing) | ✅ | ✅ |
| 22 | [Experiment Tracking](#22-experiment-tracking) | ✅ | ✅ |
| 23 | [Visualization and CSV Export](#23-visualization-and-csv-export) | ✅ | ✅ |
| 24 | [Plugin Architecture](#24-plugin-architecture) | ✅ | ❌ not exposed |
| 25 | [Benchmark Suite](#25-benchmark-suite) | ✅ | ✅ (`BenchmarkConfig`, `GABenchmark`) |
| 26 | [C API](#26-c-api) | ✅ | N/A (C only) |
| 27 | [Reproducibility Controls](#27-reproducibility-controls) | ✅ | ✅ |

**Legend:** ✅ fully available · ⚠️ partially available · ❌ not exposed in Python bindings

---

## 1. Quick Start

The fastest way to run your first optimization:

### C++

```cpp
#include <ga/genetic_algorithm.hpp>
#include <cmath>
#include <iostream>

static double sphere(const std::vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) sum += xi * xi;
    return 1000.0 / (1.0 + sum);   // maximization fitness
}

int main() {
    ga::Config cfg;
    cfg.populationSize = 50;
    cfg.generations    = 100;
    cfg.dimension      = 10;
    cfg.bounds         = {-5.12, 5.12};
    cfg.seed           = 42;

    ga::GeneticAlgorithm alg(cfg);
    ga::Result res = alg.run(sphere);

    std::cout << "Best fitness: " << res.bestFitness << "\n";
    return 0;
}
```

Build and run:

```bash
mkdir build && cd build
cmake ..
cmake --build .
./examples/ga-minimal
```

### Python

```python
import ga

def sphere(x):
    return 1000.0 / (1.0 + sum(xi**2 for xi in x))

cfg = ga.Config()
cfg.population_size = 50
cfg.generations     = 100
cfg.dimension       = 10
cfg.bounds          = ga.Bounds(-5.12, 5.12)
cfg.seed            = 42

engine = ga.GeneticAlgorithm(cfg)
result = engine.run(sphere)

print("Best fitness:", result.best_fitness)
print("Best genes:", result.best_genes)
```

---

## 2. Configuration

`ga::Config` (C++) / `ga.Config` (Python) controls all algorithm parameters.

### C++

```cpp
#include <ga/config.hpp>

ga::Config cfg;
cfg.populationSize = 60;      // number of individuals
cfg.generations    = 200;     // number of iterations
cfg.dimension      = 10;      // gene vector length
cfg.crossoverRate  = 0.8;     // probability of crossover
cfg.mutationRate   = 0.05;    // per-gene mutation probability
cfg.bounds         = {-5.12, 5.12};  // ga::Bounds{lower, upper}
cfg.eliteRatio     = 0.05;    // top 5% preserved unchanged
cfg.seed           = 42;      // 0 = random seed
```

### Python

```python
import ga

cfg = ga.Config()
cfg.population_size = 60
cfg.generations     = 200
cfg.dimension       = 10
cfg.crossover_rate  = 0.8
cfg.mutation_rate   = 0.05
cfg.bounds          = ga.Bounds(-5.12, 5.12)
cfg.elite_ratio     = 0.05
cfg.seed            = 42

print(repr(cfg))   # <Config pop=60 gen=200 dim=10>

bounds = ga.Bounds(-1.0, 1.0)
print(repr(bounds))  # Bounds(lower=-1.000000, upper=1.000000)
```

---
## 3. Chromosome Representations

The library provides eight genome types for different problem domains.

### 3.1 Real-Valued (VectorGenome\<double\>)

Default representation; used by the core `ga::GeneticAlgorithm`.

#### C++

```cpp
#include <ga/representations/vector_genome.hpp>

ga::VectorGenome<double> genome({0.1, -0.5, 0.3});
auto clone = genome.clone();
std::cout << genome.encodingName() << "\n"; // "vector<double>"
```

#### Python

`ga.VectorGenome` (real-valued) is directly constructible, and the
`ga.GeneticAlgorithm` operates on real-valued gene vectors internally.

```python
import ga

# Direct construction
g = ga.VectorGenome([0.1, -0.5, 0.3])
print(g.genes)          # [0.1, -0.5, 0.3]
print(g.encoding_name())  # "vector<double>"

# Core GA uses real-valued genes transparently
cfg = ga.Config()
cfg.dimension = 5
cfg.bounds    = ga.Bounds(0.0, 1.0)
engine = ga.GeneticAlgorithm(cfg)
result = engine.run(lambda x: sum(x))
print(result.best_genes)   # list of 5 floats
```

### 3.2 Integer (VectorGenome\<int\>)

#### C++

```cpp
#include <ga/representations/vector_genome.hpp>

ga::VectorGenome<int> genome({0, 3, 7, 2, 5});
std::cout << genome.encodingName() << "\n"; // "vector<int>"
```

#### Python

> **Not available in Python bindings.**
> `VectorGenome<int>` (integer variant) is not separately exposed to Python.
> The Python `ga.GeneticAlgorithm` operates on real-valued (`double`) gene
> vectors. As a workaround, round real-valued genes to integers inside your
> fitness function:
>
> ```python
> def int_fitness(x):
>     genes = [round(xi) for xi in x]  # convert on the fly
>     return some_objective(genes)
>
> cfg = ga.Config()
> cfg.bounds = ga.Bounds(0, 9)   # integer domain via real bounds
> engine = ga.GeneticAlgorithm(cfg)
> result = engine.run(int_fitness)
> ```

### 3.3 Binary (BitsetGenome / VectorGenome\<int\> with {0,1} bounds)

#### C++

```cpp
#include <ga/representations/vector_genome.hpp>

// Use VectorGenome<int> with bounds [0,1]
ga::VectorGenome<int> binary({0, 1, 1, 0, 1});
```

Or with the interactive demo:
```bash
echo -e "binary\nuniform\nbit_flip\ntournament" | ./build/bin/simple_ga_test
```

#### Python

`ga.BitsetGenome` is fully exposed:

```python
import ga

# Create from a list of bools
b = ga.BitsetGenome([True, False, True, True, False])
print(b.size())            # 5
print(b.popcount())        # 3

# Create zero-initialised
b2 = ga.BitsetGenome(8)    # 8 bits, all False
b2.bits[0] = True
print(b.hamming_distance(b2))

print(b.encoding_name())   # "bitset"
```

### 3.4 Permutation (PermutationGenome)

#### C++

```cpp
#include <ga/representations/vector_genome.hpp>

ga::VectorGenome<int> perm({0, 4, 2, 1, 3});  // permutation of [0..4]
```

Interactive demo for permutation problems:
```bash
echo -e "permutation\norder_crossover\nswap\ntournament" | ./build/bin/simple_ga_test
```

#### Python

`ga.PermutationGenome` is fully exposed:

```python
import ga

# From a list
p = ga.PermutationGenome([0, 4, 2, 1, 3])
print(p.size())            # 5
print(p.is_valid())        # True — valid permutation of 0..4
print(p.position_of(2))   # 2

# Random permutation (n=6, seeded)
p2 = ga.PermutationGenome.random(6, seed=42)
print(p2.order)

print(p.encoding_name())   # "permutation"
```

### 3.5 Set Genome

#### C++

```cpp
#include <ga/representations/set_genome.hpp>

ga::SetGenome<int> sg({1, 3, 5, 7});
std::cout << sg.encodingName() << "\n"; // "set<int>"
```

#### Python

`ga.SetGenome` is fully exposed:

```python
import ga

sg = ga.SetGenome({1, 3, 5, 7})
print(sg.values)           # {1, 3, 5, 7}
print(sg.encoding_name())  # "set<int>"
```

### 3.6 Map / Dictionary Genome

#### C++

ga::MapGenome<std::string, double> mg({{"x", 1.0}, {"y", -0.5}});
std::cout << mg.encodingName() << "\n"; // "map<string, double>"
```

#### Python

`ga.MapGenome` is fully exposed:

```python
import ga

mg = ga.MapGenome({"x": 1.0, "y": -0.5})
print(mg.values)            # {"x": 1.0, "y": -0.5}
print(mg.encoding_name())   # "map<string, double>"
```

### 3.7 Tree Genome (for Genetic Programming)

See [§15 Genetic Programming](#15-genetic-programming-gp).

### 3.8 N-Dimensional Array Genome

#### C++

```cpp
#include <ga/representations/ndarray_genome.hpp>

ga::NDArrayGenome<float> nd(std::vector<float>(3 * 3, 0.0f), {3, 3});
std::cout << nd.encodingName() << "\n"; // "ndarray<float>"
```

#### Python

`ga.NdArrayGenome` is fully exposed:

```python
import ga

nd = ga.NdArrayGenome(3, 3)   # 3×3 grid, initialised to 0.0
nd.set(0, 0, 1.5)
nd.set(1, 2, -0.7)
print(nd.get(0, 0))           # 1.5
print(nd.rows, nd.cols)       # 3 3
print(nd.encoding_name())     # "ndarray"
```

---

## 4. Crossover Operators

### Available Operators
| Two-Point | `TwoPointCrossover` | binary, real, integer |
| Uniform | `UniformCrossover` | binary, real, integer |
| Blend (BLX-α) | `BlendCrossover` | real |
| Simulated Binary (SBX) | `SimulatedBinaryCrossover` | real |
| Arithmetic | `IntermediateRecombination` | real |
| Line Recombination | `LineRecombination` | real |
| Order (OX) | `OrderCrossover` | permutation |
| Partially Mapped (PMX) | `PartiallyMappedCrossover` | permutation |
| Cycle (CX) | `CycleCrossover` | permutation |
| Cut-and-Crossfill | `CutAndCrossfillCrossover` | permutation |
| Edge Assembly | `EdgeCrossover` | permutation |
| Differential Evolution | `DifferentialEvolutionCrossover` | real |
| Uniform K-Vector | `UniformKVectorCrossover` | real |
| Multi-Point | `MultiPointCrossover` | binary, real |
| Subtree | `SubtreeCrossover` | tree (GP) |
| Diploid Recombination | `DiploidRecombination` | real |

### 4.1 Using crossover operators in C++

```cpp
#include <ga/genetic_algorithm.hpp>
#include "crossover/two_point_crossover.h"
#include "crossover/blend_crossover.h"
#include "crossover/order_crossover.h"

// Two-point crossover (binary/real/integer)
ga::GeneticAlgorithm alg(cfg);
alg.setCrossoverOperator(ga::makeTwoPointCrossover());

// Blend crossover (real-valued only)
alg.setCrossoverOperator(ga::makeBlendCrossover());

// Factory helpers for single use
auto xover = ga::makeOnePointCrossover();
auto [child1, child2] = xover->cross(parent1.genes, parent2.genes);
```

### 4.2 Python — exposed operators

Only **one-point** and **two-point** crossover factory functions are exposed.
All other crossover operators (Blend, SBX, Arithmetic, permutation-specific, etc.)
are not individually accessible from Python.

```python
import ga

cfg = ga.Config()
cfg.population_size = 50
cfg.generations     = 100
cfg.dimension       = 10
cfg.bounds          = ga.Bounds(-5.12, 5.12)

engine = ga.GeneticAlgorithm(cfg)

# Default: built-in crossover; override with a factory:
engine.set_crossover_operator(ga.make_one_point_crossover(seed=42))
# or
engine.set_crossover_operator(ga.make_two_point_crossover(seed=42))

result = engine.run(lambda x: 1000.0 / (1.0 + sum(xi**2 for xi in x)))
```

> **Python note:** Blend (BLX-α), SBX, Arithmetic, and all permutation-specific
> crossovers are **not available as Python factory functions**.

---

## 5. Mutation Operators

### Available Operators

| Operator | Class | Suitable For |
|----------|-------|--------------|
| Gaussian | `GaussianMutation` | real |
| Uniform | `UniformMutation` | real, integer |
| Bit-Flip | `BitFlipMutation` | binary |
| Creep | `CreepMutation` | integer |
| Random Resetting | `RandomResettingMutation` | integer |
| Swap | `SwapMutation` | permutation |
| Insert | `InsertMutation` | permutation |
| Scramble | `ScrambleMutation` | permutation |
| Inversion | `InversionMutation` | permutation |
| Self-Adaptive | `SelfAdaptiveMutation` | real (ES-style) |

### 5.1 Using mutation operators in C++

```cpp
#include <ga/genetic_algorithm.hpp>
#include "mutation/gaussian_mutation.h"
#include "mutation/swap_mutation.h"
#include "mutation/bit_flip_mutation.h"

// Gaussian mutation for real-valued
ga::GeneticAlgorithm alg(cfg);
alg.setMutationOperator(ga::makeGaussianMutation());

// Direct use
auto mut = ga::makeUniformMutation();
mut->mutate(individual.genes);
```

### 5.2 Python — exposed operators

Only **Gaussian** and **Uniform** mutation factory functions are exposed.
All other mutation operators (Bit-flip, Swap, Insert, Scramble, Inversion,
Creep, Random-resetting, Self-adaptive) are not individually accessible from Python.

```python
import ga

cfg = ga.Config()
cfg.population_size = 50
cfg.generations     = 100
cfg.dimension       = 10
cfg.bounds          = ga.Bounds(-5.12, 5.12)

engine = ga.GeneticAlgorithm(cfg)

# Gaussian mutation
engine.set_mutation_operator(ga.make_gaussian_mutation(seed=42))
# or Uniform mutation
engine.set_mutation_operator(ga.make_uniform_mutation(seed=0))

result = engine.run(lambda x: 1000.0 / (1.0 + sum(xi**2 for xi in x)))
print("Best:", result.best_fitness)
```

> **Python note:** Bit-flip, Swap, Insert, Scramble, Inversion, Creep,
> Random-resetting, and Self-adaptive mutations are
> **not available as Python factory functions**.

---

## 6. Selection Operators

### Available Operators

| Operator | Class |
|----------|-------|
| Tournament | `TournamentSelection` |
| Roulette Wheel | `RouletteWheelSelection` |
| Rank | `RankSelection` |
| Stochastic Universal Sampling (SUS) | `StochasticUniversalSampling` |
| Elitism | `ElitismSelection` |

### 6.1 Using selection operators in C++

The core engine uses its built-in tournament selection. You can integrate
selection manually when building a custom loop:

```cpp
#include "selection-operator/tournament_selection.h"
#include "selection-operator/roulette_wheel_selection.h"
#include "selection-operator/rank_selection.h"
#include "selection-operator/stochastic_universal_sampling.h"
#include "selection-operator/elitism_selection.h"

// Tournament selection (k=3)
TournamentSelection sel;
sel.setTournamentSize(3);
auto& winner = sel.select(population);

// Roulette wheel
RouletteWheelSelection rws;
auto& chosen = rws.select(population);

// Rank selection
RankSelection rs;
auto& ranked = rs.select(population);
```

### 6.2 Python

Selection strategy classes are still C++-only, but Python now exposes helper
functions that run the same selection logic over a fitness list and return the
selected indices:

- `ga.selection_tournament_indices(fitness, tournament_size=3)` (returns one index)
- `ga.selection_roulette_indices(fitness, count)`
- `ga.selection_rank_indices(fitness, count)`
- `ga.selection_sus_indices(fitness, count)`  *(stochastic universal sampling)*
- `ga.selection_elitism_indices(fitness, elite_count)`

```python
import ga

fitness = [0.1, 0.8, 0.4, 1.2, 0.6]

tournament_winner = ga.selection_tournament_indices(fitness, tournament_size=3)
roulette_picks    = ga.selection_roulette_indices(fitness, count=3)
rank_picks        = ga.selection_rank_indices(fitness, count=3)
sus_picks         = ga.selection_sus_indices(fitness, count=3)
elite_picks       = ga.selection_elitism_indices(fitness, elite_count=2)

print("Tournament winner index:", tournament_winner)
print("Roulette indices:", roulette_picks)
print("Rank indices:", rank_picks)
print("SUS indices:", sus_picks)
print("Elite indices:", elite_picks)   # tends to include the best-fitness entries
```

> `ga.GeneticAlgorithm` still uses its internal selection pipeline. These helpers
> are for analysis/custom Python loops where you need direct index selection.

---

## 7. Core GA Run and Results

### C++

```cpp
#include <ga/genetic_algorithm.hpp>
#include <iostream>

double rastrigin(const std::vector<double>& x) {
    const double A = 10.0;
    double sum = A * x.size();
    for (double xi : x) sum += xi*xi - A*std::cos(2*M_PI*xi);
    return 1000.0 / (1.0 + sum);
}

int main() {
    ga::Config cfg;
    cfg.populationSize = 60;
    cfg.generations    = 200;
    cfg.dimension      = 10;
    cfg.bounds         = {-5.12, 5.12};
    cfg.seed           = 1;

    ga::GeneticAlgorithm alg(cfg);
    ga::Result res = alg.run(rastrigin);

    std::cout << "Best fitness: "  << res.bestFitness << "\n";
    std::cout << "Best genes[0]: " << res.bestGenes[0] << "\n";

    // Convergence history
    for (size_t i = 0; i < res.bestHistory.size(); ++i)
        std::cout << "Gen " << i << ": best=" << res.bestHistory[i]
                  << " avg=" << res.avgHistory[i] << "\n";
    return 0;
}
```

### Python

```python
import ga
import math

def rastrigin(x):
    A = 10.0
    val = A * len(x) + sum(xi**2 - A * math.cos(2 * math.pi * xi) for xi in x)
    return 1000.0 / (1.0 + val)

cfg = ga.Config()
cfg.population_size = 60
cfg.generations     = 200
cfg.dimension       = 10
cfg.bounds          = ga.Bounds(-5.12, 5.12)
cfg.seed            = 1

engine = ga.GeneticAlgorithm(cfg)
result = engine.run(rastrigin)

print("Best fitness:", result.best_fitness)
print("Best genes:",   result.best_genes)

# Convergence history
for gen, (best, avg) in enumerate(zip(result.best_history, result.avg_history)):
    if gen % 50 == 0:
        print(f"Gen {gen:3d}: best={best:.4f}  avg={avg:.4f}")
```

---

## 8. High-Level Optimizer API

A convenience API for common workflows.

### C++

```cpp
#include <ga/api/optimizer.hpp>
#include <iostream>

int main() {
    ga::Config cfg;
    cfg.populationSize = 80;
    cfg.generations    = 300;
    cfg.dimension      = 10;
    cfg.bounds         = {-5.12, 5.12};
    cfg.seed           = 42;

    auto result = ga::api::Optimizer()
        .withConfig(cfg)
        .withSeed(42)
        .optimize([](const std::vector<double>& x) {
            double s = 0; for (double xi : x) s += xi*xi;
            return 1000.0 / (1.0 + s);
        });

    std::cout << "Best: " << result.bestFitness << "\n";
    return 0;
}
```

Multi-objective variant:

```cpp
#include <ga/api/optimizer.hpp>
#include <iostream>

int main() {
    ga::Config cfg;
    cfg.populationSize = 100;
    cfg.dimension      = 2;
    cfg.bounds         = {-5.0, 5.0};
    cfg.seed           = 0;

    // objectives: each is a callable returning a scalar to maximise
    std::vector<ga::api::Optimizer::Objective> objectives = {
        [](const std::vector<double>& x) { return -x[0]*x[0]; },
        [](const std::vector<double>& x) { return -(x[0]-2)*(x[0]-2); },
    };

    auto moResult = ga::api::Optimizer()
        .withConfig(cfg)
        .optimizeMultiObjective(objectives, /*popSize=*/100, /*gens=*/200);

    std::cout << "Pareto front size: " << moResult.paretoGenes.size() << "\n";
    return 0;
}
```

### Python

Both `ga.Optimizer` (fluent facade) and `ga.OptimizerBuilder` are exposed.

```python
import ga

# --- Fluent Optimizer ---
cfg = ga.Config()
cfg.population_size = 80
cfg.generations     = 300
cfg.dimension       = 10
cfg.bounds          = ga.Bounds(-5.12, 5.12)
cfg.seed            = 42

result = (ga.Optimizer()
    .with_config(cfg)
    .with_seed(42)
    .optimize(lambda x: 1000.0 / (1.0 + sum(xi**2 for xi in x))))

print("Best:", result.best_fitness)

# --- Multi-objective via Optimizer ---
objectives = [
    lambda x: -x[0]*x[0],
    lambda x: -(x[0]-2)*(x[0]-2),
]
mo_result = (ga.Optimizer()
    .with_config(cfg)
    .optimize_multi_objective_nsga2(objectives, population_size=100, generations=200))
print("Pareto front size:", len(mo_result.pareto_genes))

# --- Builder API ---
builder = (ga.OptimizerBuilder()
    .dimension(10)
    .bounds(-5.12, 5.12)
    .population_size(80)
    .generations(300)
    .seed(42))

opt = builder.build()
result2 = opt.optimize(lambda x: 1000.0 / (1.0 + sum(xi**2 for xi in x)))
print("Builder best:", result2.best_fitness)
```

---

## 9. Multi-Objective: NSGA-II

### 9.1 C++ — full run loop

```cpp
#include <ga/algorithms/moea/nsga2.hpp>
#include <ga/core/individual.hpp>
#include <iostream>
#include <random>

int main() {
    ga::moea::Nsga2Config cfg;
    cfg.populationSize = 100;
    cfg.generations    = 200;
    cfg.seed           = 42;

    ga::moea::Nsga2 nsga2(cfg);

    // Initialize population with random 2D gene vectors
    std::mt19937 rng(cfg.seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<ga::Individual> initPop(cfg.populationSize);
    for (auto& ind : initPop) {
        ind.evaluation.objectives = {dist(rng), dist(rng)};
    }

    // Evaluate: assign 2-objective values to each individual
    auto evaluate = [](std::vector<ga::Individual>& pop) {
        for (auto& ind : pop) {
            const double x = ind.evaluation.objectives.empty() ? 0.0
                             : ind.evaluation.objectives[0];
            ind.evaluation.objectives = {x*x, (x-2)*(x-2)};
        }
    };

    // Reproduce: simple uniform crossover stub
    auto reproduce = [](const std::vector<ga::Individual>& parents,
                        const std::vector<std::size_t>& selected,
                        std::mt19937& r) -> std::vector<ga::Individual> {
        std::vector<ga::Individual> offspring;
        offspring.reserve(selected.size());
        for (auto idx : selected) {
            offspring.push_back(parents[idx % parents.size()]);
        }
        return offspring;
    };

    auto result = nsga2.run(initPop, evaluate, reproduce);
    std::cout << "Final population size: " << result.population.size() << "\n";
    std::cout << "Generations tracked:   " << result.history.size() << "\n";
    return 0;
}
```

### 9.2 C++ — objective-space utilities

```cpp
#include <ga/algorithms/moea/nsga2.hpp>

ga::moea::Nsga2 nsga2;
std::vector<ga::Individual> pop = /* ... build population ... */;

// Non-dominated sorting → returns vector of front indices
auto fronts = nsga2.nonDominatedSort(pop);

// Crowding distance for the first front
auto distances = nsga2.crowdingDistance(pop, fronts[0]);
```

### 9.3 Python — objective-space utilities

```python
import ga

# Sample objective vectors (2 objectives, minimization)
objectives = [
    [0.1, 0.9],
    [0.4, 0.5],
    [0.9, 0.1],
    [0.5, 0.5],
    [0.2, 0.8],
]

# --- Convenience functions ---
fronts = ga.nsga2_non_dominated_sort(objectives)
print("Front 0 indices:", fronts[0])

front0_distances = ga.nsga2_crowding_distance(objectives, fronts[0])
print("Crowding distances:", front0_distances)

# --- Class-based API ---
cfg = ga.Nsga2Config()
cfg.population_size = 100
cfg.generations     = 200
cfg.seed            = 42

nsga2 = ga.Nsga2(cfg)
fronts2 = nsga2.non_dominated_sort_objectives(objectives)
dist2   = nsga2.crowding_distance_objectives(objectives, fronts2[0])
```

> **Python note:** The full NSGA-II *run loop* is not exposed. Only the
> objective-space utilities (sorting, crowding distance) are available.

---

## 10. Multi-Objective: NSGA-III

### 10.1 C++ — reference points and selection

```cpp
#include <ga/moea/nsga3.hpp>
#include <iostream>

int main() {
    // Generate Das-Dennis reference points for 3 objectives, 4 divisions
    auto refs = ga::moea::Nsga3::generateDasDennisReferencePoints(3, 4);
    std::cout << "Reference point count: " << refs.size() << "\n";

    ga::moea::Nsga2Config cfg;
    cfg.populationSize = 92;   // ≥ number of reference points
    cfg.generations    = 300;

    ga::moea::Nsga3 nsga3(cfg);

    // Run environmental selection on a population
    std::vector<ga::Individual> pop = /* ... */;
    auto selected = nsga3.environmentalSelect(pop, 91, refs);
    return 0;
}
```

### 10.2 Python — reference points and selection

```python
import ga

# Generate 15 reference points (3 objectives, 4 divisions)
refs = ga.nsga3_reference_points(3, 4)
print(f"Reference points: {len(refs)}")    # 15

# Sample 5 objective vectors (3-objective, minimization)
objectives = [
    [0.1, 0.5, 0.4],
    [0.5, 0.1, 0.4],
    [0.4, 0.4, 0.2],
    [0.2, 0.6, 0.2],
    [0.7, 0.1, 0.2],
]

# Convenience: environmental selection (returns surviving indices)
selected_idx = ga.nsga3_environmental_select_indices(objectives, 3, refs)
print("Selected:", selected_idx)

# Class-based API
cfg = ga.Nsga2Config()
cfg.population_size = 5
cfg.seed = 0

nsga3 = ga.Nsga3(cfg)

# Non-dominated sorting
fronts = nsga3.non_dominated_sort_objectives(objectives)
print("Front 0:", fronts[0])

# Environmental selection (returns surviving objective vectors)
surviving = nsga3.environmental_select_objectives(objectives, 3, refs)

# Environmental selection (returns surviving indices)
indices = nsga3.environmental_select_indices(objectives, 3, refs)
print("Surviving indices:", indices)
```

---

## 11. Multi-Objective: SPEA2

Strength Pareto Evolutionary Algorithm 2.

### C++

```cpp
#include <ga/moea/spea2.hpp>
#include <ga/core/individual.hpp>
#include <iostream>

int main() {
    // Build a population with 2-objective evaluations
    std::vector<ga::Individual> pop(4);
    pop[0].evaluation.objectives = {0.1, 0.9};
    pop[1].evaluation.objectives = {0.5, 0.5};
    pop[2].evaluation.objectives = {0.9, 0.1};
    pop[3].evaluation.objectives = {0.4, 0.6};

    ga::moea::Spea2 spea2;

    // Compute strength-based fitness values (lower is better)
    auto fit = spea2.strengthFitness(pop);
    for (std::size_t i = 0; i < fit.size(); ++i)
        std::cout << "ind " << i << " SPEA2 fitness: " << fit[i] << "\n";

    // Select best `archiveSize` individuals
    auto archive = spea2.environmentalSelect(pop, /*archiveSize=*/2);
    std::cout << "Archive size: " << archive.size() << "\n";
    return 0;
}
```

### Python

`ga.Spea2` objective-space utilities are fully exposed:

```python
import ga

# Sample 4 objective vectors (2 objectives, minimization)
objectives = [
    [0.1, 0.9],
    [0.5, 0.5],
    [0.9, 0.1],
    [0.4, 0.6],
]

spea2 = ga.Spea2()

# Compute SPEA2 strength fitness (lower is better)
fitness = spea2.strength_fitness_objectives(objectives)
print("SPEA2 fitness values:", fitness)

# Environmental selection — returns surviving objective vectors
archive = spea2.environmental_select_objectives(objectives, target_size=2)
print("Archive size:", len(archive))

# Environmental selection — returns surviving indices
indices = spea2.environmental_select_indices(objectives, target_size=2)
print("Archive indices:", indices)

# --- Convenience functions ---
fitness2  = ga.spea2_strength_fitness(objectives)
indices2  = ga.spea2_environmental_select_indices(objectives, target_size=2)
```

---

## 12. Multi-Objective: MO-CMA-ES

Multi-Objective CMA-ES wrapper.

### C++

```cpp
#include <ga/moea/mo_cmaes.hpp>

ga::moea::MoCmaEs moea;
// MO-CMA-ES uses the same environmental-selection interface as NSGA-II.
// Configure via ga::moea::Nsga2Config and supply an evaluation callback.
```

### Python

`ga.MoCmaEs` is fully exposed:

```python
import ga

cma_cfg = ga.CmaEsConfig()
cma_cfg.dimension       = 2
cma_cfg.population_size = 20
cma_cfg.generations     = 200
cma_cfg.sigma           = 0.5
cma_cfg.lower           = -5.0
cma_cfg.upper           =  5.0

mo_cfg = ga.MoCmaEsConfig()
mo_cfg.cma     = cma_cfg
mo_cfg.weights = [0.5, 0.5]   # equal weight for each objective

moea = ga.MoCmaEs(mo_cfg)

def sphere2(x):
    return x[0]**2 + x[1]**2

result = moea.run(sphere2)
print("MO-CMA-ES best:", result.best)
print("Objectives:", result.objectives)
print("Weighted fitness:", result.weighted_fitness)
```

---

## 13. Evolution Strategies (ES)

`(μ,λ)` and `(μ+λ)` strategies.

### C++

```cpp
#include <ga/es/evolution_strategies.hpp>
#include <iostream>

int main() {
    ga::es::EvolutionStrategyConfig cfg;
    cfg.mu          = 10;     // parents
    cfg.lambda      = 50;     // offspring (struct member — not a keyword conflict)
    cfg.dimension   = 5;
    cfg.sigma       = 0.3;    // initial step size
    cfg.generations = 200;
    cfg.plusStrategy = false; // false=(mu,lambda), true=(mu+lambda)
    cfg.lower       = -5.0;
    cfg.upper       =  5.0;

    ga::es::EvolutionStrategy es(cfg);

    auto result = es.run([](const std::vector<double>& x) {
        double s = 0; for (double xi : x) s += xi * xi;
        return -s;   // ES minimises; return negative value to maximise
    });

    std::cout << "Best: " << result.bestFitness << "\n";
    return 0;
}
```

### Python

`ga.EvolutionStrategy` is fully exposed:

```python
import ga

cfg = ga.EvolutionStrategyConfig()
cfg.mu           = 10     # parents
cfg.lambda_      = 50     # offspring
cfg.dimension    = 5
cfg.sigma        = 0.3    # initial step size
cfg.generations  = 200
cfg.plus_strategy = False  # False=(mu,lambda), True=(mu+lambda)
cfg.lower        = -5.0
cfg.upper        =  5.0

es = ga.EvolutionStrategy(cfg)

result = es.run(lambda x: -sum(xi**2 for xi in x))  # maximise negated sphere

print("Best fitness:", result.best_fitness)
print("Best solution:", result.best)

# Convergence history
for gen, f in enumerate(result.best_history[::50]):
    print(f"  Gen {gen*50}: {f:.4f}")
```

---

## 14. CMA-ES

Covariance Matrix Adaptation Evolution Strategy (diagonal variant).

### C++

```cpp
#include <ga/es/cmaes.hpp>
#include <iostream>

int main() {
    ga::es::CmaEsConfig cfg;
    cfg.dimension      = 10;
    cfg.populationSize = 20;  // offspring count
    cfg.sigma          = 0.5;
    cfg.generations    = 500;
    cfg.lower          = -5.0;
    cfg.upper          =  5.0;

    ga::es::DiagonalCmaEs cmaes(cfg);

    auto result = cmaes.run([](const std::vector<double>& x) {
        double s = 0; for (double xi : x) s += xi * xi;
        return -s;
    });

    std::cout << "Best objective: " << -result.bestFitness << "\n";
    return 0;
}
```

### Python

`ga.DiagonalCmaEs` is fully exposed:

```python
import ga

cfg = ga.CmaEsConfig()
cfg.dimension       = 10
cfg.population_size = 20   # offspring count
cfg.sigma           = 0.5
cfg.generations     = 500
cfg.lower           = -5.0
cfg.upper           =  5.0

cmaes = ga.DiagonalCmaEs(cfg)

result = cmaes.run(lambda x: -sum(xi**2 for xi in x))  # maximise negated sphere

print("Best objective (sphere):", -result.best_fitness)
print("Best solution:", result.best)
print("History length:", len(result.history))
```

---

## 15. Genetic Programming (GP)

Typed tree-based GP with Automatically Defined Functions (ADF).

### 15.1 Tree nodes and builder (C++)

```cpp
#include <ga/gp/node.hpp>
#include <ga/gp/type_system.hpp>
#include <ga/gp/tree_builder.hpp>
#include <ga/gp/adf.hpp>
#include <iostream>
#include <random>

int main() {
    // Define primitives (function nodes and terminal nodes)
    std::vector<ga::gp::Primitive> primitives;

    // Function: "+" with 2 Double arguments → Double
    ga::gp::Primitive plus;
    plus.name = "+"; plus.isTerminal = false;
    plus.signature.returnType = ga::gp::ValueType::Double;
    plus.signature.argTypes   = {ga::gp::ValueType::Double, ga::gp::ValueType::Double};
    primitives.push_back(plus);

    // Terminal: "x"
    ga::gp::Primitive xTerm;
    xTerm.name = "x"; xTerm.isTerminal = true;
    xTerm.signature.returnType = ga::gp::ValueType::Double;
    primitives.push_back(xTerm);

    // Terminal: constant "1.0"
    ga::gp::Primitive c1;
    c1.name = "1.0"; c1.isTerminal = true;
    c1.signature.returnType = ga::gp::ValueType::Double;
    primitives.push_back(c1);

    ga::gp::TreeBuilder builder(primitives);

    std::mt19937 rng(42);
    auto tree = builder.grow(/*maxDepth=*/3,
                             ga::gp::ValueType::Double,
                             /*stronglyTyped=*/true,
                             rng);
    std::cout << "Tree root: " << tree->symbol
              << " size=" << tree->size() << "\n";

    // ADF pool: store a named sub-function
    ga::gp::ADFPool pool;
    pool.put("adf0", *tree);
    std::cout << "ADF pool size: " << pool.size() << "\n";
    const ga::gp::Node& adf = pool.get("adf0");
    std::cout << "Retrieved ADF root: " << adf.symbol << "\n";

    return 0;
}
```

### 15.2 Python — GP primitives, tree builder, ADF pool

`ga.ValueType`, `ga.Signature`, `ga.Primitive`, `ga.Node`, `ga.TreeGenome`,
`ga.TreeBuilder`, and `ga.ADFPool` are all fully exposed:

```python
import ga

# --- Define primitives ---
plus = ga.Primitive()
plus.name       = "+"
plus.is_terminal = False
plus.signature.return_type = ga.ValueType.double
plus.signature.arg_types   = [ga.ValueType.double, ga.ValueType.double]

x_term = ga.Primitive()
x_term.name       = "x"
x_term.is_terminal = True
x_term.signature.return_type = ga.ValueType.double

c1 = ga.Primitive()
c1.name       = "1.0"
c1.is_terminal = True
c1.signature.return_type = ga.ValueType.double

# --- Build a random tree ---
builder = ga.TreeBuilder([plus, x_term, c1])
tree = builder.grow(max_depth=3,
                    target_type=ga.ValueType.double,
                    strongly_typed=True,
                    seed=42)
print("Root symbol:", tree.symbol)
print("Tree size:  ", tree.size())

# --- TreeGenome ---
genome = ga.TreeGenome(tree)
print("Has root:", genome.has_root())
print("Encoding:", genome.encoding_name())

# --- ADF pool ---
pool = ga.ADFPool()
pool.put("adf0", tree)
print("ADF pool size:", pool.size())
print("ADF has 'adf0':", pool.has("adf0"))
adf_root = pool.get("adf0")
print("Retrieved ADF root:", adf_root.symbol)
```

---

## 16. Adaptive Operators

Dynamic mutation/crossover rate controller based on diversity and progress metrics.

### C++

```cpp
#include <ga/adaptive/adaptive_policy.hpp>
#include <iostream>

int main() {
    // minMutation, maxMutation, minCrossover, maxCrossover
    ga::adaptive::AdaptiveRateController controller(0.01, 0.30, 0.50, 0.95);

    ga::adaptive::AdaptiveRates rates;
    rates.mutationRate  = 0.05;
    rates.crossoverRate = 0.80;

    double diversity    = 0.10;  // low diversity → increase mutation
    double improvement  = 0.0;   // no improvement → increase mutation

    auto adapted = controller.update(rates, diversity, improvement);

    std::cout << "Adapted mutation rate:   " << adapted.mutationRate  << "\n";
    std::cout << "Adapted crossover rate:  " << adapted.crossoverRate << "\n";
    return 0;
}
```

### Python

`ga.AdaptiveRates` and `ga.AdaptiveRateController` are fully exposed:

```python
import ga

# min_mutation, max_mutation, min_crossover, max_crossover
controller = ga.AdaptiveRateController(0.01, 0.30, 0.50, 0.95)

rates = ga.AdaptiveRates()
rates.mutation_rate  = 0.05
rates.crossover_rate = 0.80

diversity   = 0.10   # low diversity → increase mutation
improvement = 0.0    # no improvement → increase mutation

adapted = controller.update(rates, diversity, improvement)
print("Adapted mutation rate: ", adapted.mutation_rate)
print("Adapted crossover rate:", adapted.crossover_rate)
```

---

## 17. Hybrid Optimization

GA combined with a local search (memetic algorithm pattern).

### C++

```cpp
#include <ga/hybrid/hybrid_optimizer.hpp>
#include <iostream>
#include <algorithm>

static double sphere(const std::vector<double>& x) {
    double s = 0; for (double xi : x) s += xi*xi;
    return 1000.0 / (1.0 + s);
}

int main() {
    ga::Config cfg;
    cfg.populationSize = 50;
    cfg.generations    = 100;
    cfg.dimension      = 5;
    cfg.bounds         = {-5.0, 5.0};

    // Local search: hill-climb by nudging each gene
    auto localSearch = [](std::vector<double>& genes) {
        for (double& g : genes)
            g = std::max(-5.0, std::min(5.0, g * 0.99));  // simple shrink step
    };

    ga::hybrid::HybridOptimizer opt(cfg);
    auto result = opt.run(sphere, localSearch, /*localSearchRestarts=*/10);

    std::cout << "Hybrid best: " << result.bestFitness << "\n";
    return 0;
}
```

### Python

`ga.HybridOptimizer` is fully exposed:

```python
import ga

cfg = ga.Config()
cfg.population_size = 50
cfg.generations     = 100
cfg.dimension       = 5
cfg.bounds          = ga.Bounds(-5.0, 5.0)

def sphere(x):
    return 1000.0 / (1.0 + sum(xi**2 for xi in x))

# Optional local search callable (list[float] -> list[float] or None)
def local_search(genes):
    return [max(-5.0, min(5.0, g * 0.99)) for g in genes]

opt = ga.HybridOptimizer(cfg)
result = opt.run(sphere, local_search=local_search, local_search_restarts=10)
print("Hybrid best:", result.best_fitness)
```

---

## 18. Constraint Handling

Hard constraints, soft penalty functions, and repair operators.

### C++

```cpp
#include <ga/constraints/constraints.hpp>
#include <iostream>
#include <algorithm>

int main() {
    ga::constraints::ConstraintSet cs;

    // Hard constraint: x[0] + x[1] <= 1.0
    cs.hard.push_back([](const std::vector<double>& x) {
        return x[0] + x[1] <= 1.0;
    });

    // Soft penalty: penalise violation of x[0] >= 0
    cs.soft.push_back([](const std::vector<double>& x) -> double {
        return (x[0] < 0) ? -x[0] * 100.0 : 0.0;
    });

    // Repair: clamp each gene to [0, inf)
    cs.repairs.push_back([](std::vector<double>& x) {
        for (double& xi : x) xi = std::max(0.0, xi);
    });

    std::vector<double> genes = {0.5, 0.8};

    bool feasible = ga::constraints::isFeasible(genes, cs);
    double penalty = ga::constraints::totalPenalty(genes, cs);
    ga::constraints::applyRepairs(genes, cs);

    std::cout << "Feasible: " << feasible << "  Penalty: " << penalty << "\n";

    // Convenience: combine base fitness with feasibility penalty
    double base = 500.0;
    double adjusted = ga::constraints::penalizedFitness(base, genes, cs, /*infeasiblePenalty=*/1e6);
    std::cout << "Adjusted fitness: " << adjusted << "\n";
    return 0;
}
```

### Python

`ga.ConstraintSet` and related helpers are fully exposed:

```python
import ga

cs = ga.ConstraintSet()

# Hard constraint: x[0] + x[1] <= 1.0
cs.add_hard_constraint(lambda x: x[0] + x[1] <= 1.0)

# Soft penalty: penalise x[0] < 0
cs.add_soft_penalty(lambda x: max(0.0, -x[0]) * 100.0)

# Repair: clamp each gene to [0, inf)
cs.add_repair(lambda x: [max(0.0, xi) for xi in x])

genes = [0.5, 0.8]
print("Feasible:", ga.is_feasible(genes, cs))
print("Penalty:", ga.total_penalty(genes, cs))

repaired = ga.apply_repairs(genes, cs)
print("Repaired genes:", repaired)

base_fitness = 500.0
adjusted = ga.penalized_fitness(base_fitness, genes, cs, infeasible_penalty=1e6)
print("Adjusted fitness:", adjusted)
```

---

## 19. Parallel and Distributed Evaluation

Thread-pool and process-based parallel fitness evaluation.

### 19.1 Parallel evaluator (C++)

```cpp
#include <ga/evaluation/parallel_evaluator.hpp>
#include <ga/genetic_algorithm.hpp>

int main() {
    // Use ParallelEvaluator directly to batch-evaluate a list of gene vectors
    auto fitnessFn = [](const std::vector<double>& x) {
        double s = 0; for (double xi : x) s += xi*xi; return 1000.0/(1.0+s);
    };

    ga::evaluation::ParallelEvaluator<std::vector<double>, double, decltype(fitnessFn)>
        evaluator(fitnessFn, /*threads=*/4);

    // Build a batch of candidate gene vectors
    std::vector<std::vector<double>> batch = {
        {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}, {0.7, 0.8}
    };
    auto results = evaluator.evaluate(batch);
    for (double r : results) std::cout << r << "\n";
    return 0;
}
```

### 19.2 Local distributed executor (C++)

```cpp
#include <ga/evaluation/distributed_executor.hpp>
#include <iostream>

int main() {
    // Threaded local backend wrapping a fitness function
    ga::evaluation::LocalDistributedExecutor exec(
        [](const std::vector<double>& x) {
            double s = 0; for (double xi : x) s += xi*xi; return 1000.0/(1.0+s);
        },
        /*workers=*/4
    );

    std::vector<std::vector<double>> batch = {{0.1, 0.2}, {0.3, 0.4}};
    auto results = exec.execute(batch);
    for (double r : results) std::cout << r << "\n";
    return 0;
}

### 19.3 Process distributed executor (POSIX, C++)

```cpp
#include <ga/evaluation/distributed_executor.hpp>
#include <iostream>

int main() {
    // Multi-process backend (POSIX fork) — Linux/macOS only
    ga::evaluation::ProcessDistributedExecutor pexec(
        [](const std::vector<double>& x) {
            double s = 0; for (double xi : x) s += xi*xi; return 1000.0/(1.0+s);
        },
        /*workers=*/4
    );

    std::vector<std::vector<double>> batch = {{0.1, 0.2}, {0.3, 0.4}};
    auto results = pexec.execute(batch);
    for (double r : results) std::cout << r << "\n";
    return 0;
}

### Python

Python exposes thread-parallel evaluators directly:

- `ga.ParallelEvaluator(fitness, threads=...)`
- `ga.LocalDistributedExecutor(evaluator, workers=...)`
- plus optimizer-level threading via `ga.Optimizer.with_threads(...)`

```python
import ga

def sphere(x):
    return 1000.0 / (1.0 + sum(xi * xi for xi in x))

batch = [[0.1, 0.2], [0.3, 0.4], [0.0, 0.0]]

pe = ga.ParallelEvaluator(sphere, threads=4)
print("ParallelEvaluator:", pe.evaluate(batch))

lde = ga.LocalDistributedExecutor(sphere, workers=4)
print("LocalDistributedExecutor:", lde.execute(batch))

cfg = ga.Config()
cfg.population_size = 120
cfg.generations = 200
cfg.dimension = 20
cfg.bounds = ga.Bounds(-5.12, 5.12)
result = (ga.Optimizer()
    .with_config(cfg)
    .with_threads(4)
    .with_seed(42)
    .optimize(sphere))
print("Best fitness:", result.best_fitness)
```

> `ProcessDistributedExecutor` is still C++-only (POSIX/fork backend).

---

## 20. Co-Evolution

Multi-population cooperative / competitive co-evolution.

### C++

```cpp
#include <ga/coevolution/coevolution.hpp>
#include <ga/core/individual.hpp>
#include <iostream>

int main() {
    ga::coevolution::CoevolutionConfig cfg;
    cfg.generations = 100;
    cfg.seed        = 42;

    ga::coevolution::CoevolutionEngine engine(cfg);

    // Build two populations
    using Pops = ga::coevolution::CoevolutionEngine::Populations;
    Pops populations(2);
    for (auto& pop : populations) {
        pop.resize(20);  // 20 individuals per population
    }

    auto result = engine.run(
        populations,
        // Evaluate: compute fitness for each population (e.g., predator-prey)
        [](Pops& pops) {
            for (auto& pop : pops)
                for (auto& ind : pop)
                    ind.evaluation.objectives = {1.0};  // placeholder
        },
        // Reproduce: apply variation within each population
        [](Pops& pops, std::mt19937& /*rng*/) {
            /* apply crossover / mutation within each pop */
        }
    );

    std::cout << "Populations: " << result.size() << "\n";
    return 0;
}
```

### Python

`ga.CoevolutionConfig` and `ga.CoevolutionEngine` are fully exposed:

```python
import ga

cfg = ga.CoevolutionConfig()
cfg.generations = 100
cfg.seed        = 42

engine = ga.CoevolutionEngine(cfg)

# Build two populations (list of list-of-Individual)
pop1 = [ga.Individual() for _ in range(20)]
pop2 = [ga.Individual() for _ in range(20)]
for ind in pop1 + pop2:
    ind.evaluation.objectives = [1.0]  # placeholder

def evaluate(pops):
    for pop in pops:
        for ind in pop:
            ind.evaluation.objectives = [1.0]  # custom evaluation logic

def reproduce(pops):
    pass  # custom reproduction logic

result = engine.run([pop1, pop2], evaluate=evaluate, reproduce=reproduce)
print("Populations returned:", len(result))
```

---

## 21. Checkpointing

Save and restore full run state (binary or JSON).

### C++

```cpp
#include <ga/checkpoint/checkpoint.hpp>
#include <ga/genetic_algorithm.hpp>

int main() {
    ga::Config cfg;
    cfg.populationSize = 50;
    cfg.generations    = 200;
    cfg.dimension      = 10;
    cfg.bounds         = {-5.12, 5.12};
    cfg.seed           = 7;

    ga::GeneticAlgorithm alg(cfg);
    ga::Result result = alg.run([](const std::vector<double>& x){
        double s=0; for(auto xi:x) s+=xi*xi; return 1000.0/(1.0+s);
    });

    // Save
    ga::checkpoint::CheckpointState state;
    state.config     = cfg;
    state.result     = result;
    state.generation = cfg.generations - 1;
    state.rngState   = "serialized_rng";

    ga::checkpoint::CheckpointManager::saveJson("run.json", state);
    ga::checkpoint::CheckpointManager::saveBinary("run.bin", state);

    // Load
    auto loaded = ga::checkpoint::CheckpointManager::loadJson("run.json");
    std::cout << "Loaded generation: " << loaded.generation << "\n";
    std::cout << "Loaded best:       " << loaded.result.bestFitness << "\n";
    return 0;
}
```

### Python

```python
import ga

# --- Build state from a completed run ---
cfg = ga.Config()
cfg.population_size = 50
cfg.generations     = 200
cfg.dimension       = 10
cfg.bounds          = ga.Bounds(-5.12, 5.12)
cfg.seed            = 7

engine = ga.GeneticAlgorithm(cfg)
result = engine.run(lambda x: 1000.0 / (1.0 + sum(xi**2 for xi in x)))

state = ga.CheckpointState()
state.config     = cfg
state.result     = result
state.generation = cfg.generations - 1
state.rng_state  = "py-run"

# Save to JSON
ga.checkpoint_save_json("run.json", state)

# Save to binary
ga.checkpoint_save_binary("run.bin", state)

# Reload from JSON
loaded = ga.checkpoint_load_json("run.json")
print("Generation:", loaded.generation)
print("Best fitness:", loaded.result.best_fitness)

# Reload from binary
loaded_bin = ga.checkpoint_load_binary("run.bin")
print("Binary generation:", loaded_bin.generation)
```

> **Note:** Both JSON and binary checkpoints are exposed in Python.
> `ga.checkpoint_save_binary` / `ga.checkpoint_load_binary` are available
> alongside the JSON variants.

---

## 22. Experiment Tracking

Logs run metadata, per-generation metrics, and best solutions for reproducibility.

### C++

```cpp
#include <ga/tracking/experiment_tracker.hpp>
#include <ga/genetic_algorithm.hpp>

int main() {
    ga::Config cfg;
    cfg.populationSize = 60;
    cfg.generations    = 100;
    cfg.dimension      = 5;
    cfg.bounds         = {-5.0, 5.0};
    cfg.seed           = 99;

    ga::tracking::ExperimentTracker tracker("experiment_001");
    tracker.writeConfig(cfg, "experiment_001_config.txt");

    ga::GeneticAlgorithm alg(cfg);
    ga::Result result = alg.run([](const std::vector<double>& x){
        double s=0; for(auto xi:x) s+=xi*xi; return 1000.0/(1.0+s);
    });

    tracker.writeHistoryCSV(result, "experiment_001_history.csv");
    tracker.writeBestSolutionCSV(result, "experiment_001_best.csv");
    return 0;
}
```

### Python

`ga.ExperimentTracker` is fully exposed:

```python
import ga

cfg = ga.Config()
cfg.population_size = 60
cfg.generations     = 100
cfg.dimension       = 5
cfg.bounds          = ga.Bounds(-5.0, 5.0)
cfg.seed            = 99

tracker = ga.ExperimentTracker("experiment_001")
tracker.write_config(cfg, "experiment_001_config.txt")

engine = ga.GeneticAlgorithm(cfg)
result = engine.run(lambda x: 1000.0 / (1.0 + sum(xi**2 for xi in x)))

tracker.write_history_csv(result, "experiment_001_history.csv")
tracker.write_best_solution_csv(result, "experiment_001_best.csv")
```

---

## 23. Visualization and CSV Export

Export fitness curves, Pareto fronts, and diversity trends to CSV/JSON.

### C++

```cpp
#include <ga/visualization/export.hpp>
#include <ga/genetic_algorithm.hpp>

int main() {
    ga::Config cfg;
    cfg.populationSize = 50;
    cfg.generations    = 100;
    cfg.dimension      = 5;
    cfg.bounds         = {-5.0, 5.0};

    ga::GeneticAlgorithm alg(cfg);
    ga::Result result = alg.run([](const std::vector<double>& x){
        double s=0; for(auto xi:x) s+=xi*xi; return 1000.0/(1.0+s);
    });

    // Export fitness curve (generation, best, avg)
    ga::visualization::exportFitnessCurveCSV(
        result.bestHistory, result.avgHistory, "fitness.csv");

    // Export diversity trend (pass your own per-generation diversity values)
    std::vector<double> diversity(result.bestHistory.size(), 0.5); // placeholder
    ga::visualization::exportDiversityCSV(diversity, "diversity.csv");

    // Export Pareto front (for multi-objective runs)
    std::vector<std::vector<double>> pareto = {{0.1, 0.9}, {0.5, 0.5}, {0.9, 0.1}};
    ga::visualization::exportParetoFrontCSV(pareto, "pareto.csv");
    return 0;
}
```

### Python

`ga.export_fitness_curve_csv`, `ga.export_pareto_front_csv`, and
`ga.export_diversity_csv` are fully exposed:

```python
import ga

cfg = ga.Config()
cfg.population_size = 50
cfg.generations     = 100
cfg.dimension       = 5
cfg.bounds          = ga.Bounds(-5.0, 5.0)

engine = ga.GeneticAlgorithm(cfg)
result = engine.run(lambda x: 1000.0 / (1.0 + sum(xi**2 for xi in x)))

# Export fitness curve (generation, best, avg)
ga.export_fitness_curve_csv(result.best_history, result.avg_history, "fitness.csv")
print("Exported fitness.csv")

# Export diversity trend (supply your own per-generation values)
diversity = [0.5] * len(result.best_history)
ga.export_diversity_csv(diversity, "diversity.csv")

# Export Pareto front (for multi-objective runs)
pareto = [[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]
ga.export_pareto_front_csv(pareto, "pareto.csv")
```

---

## 24. Plugin Architecture

Register custom operators at runtime by string key.

### C++

```cpp
#include <ga/plugin/registry.hpp>
#include "crossover/base_crossover.h"
#include <memory>
#include <iostream>

// Custom crossover implementing the standard interface
class MyXover : public BaseCrossover {
public:
    std::pair<std::vector<double>, std::vector<double>>
    cross(const std::vector<double>& p1, const std::vector<double>& p2) override {
        // simple copy swap at midpoint
        size_t mid = p1.size() / 2;
        auto c1 = p1; auto c2 = p2;
        for (size_t i = mid; i < p1.size(); ++i) std::swap(c1[i], c2[i]);
        return {c1, c2};
    }
};

int main() {
    ga::plugin::Registry<BaseCrossover> xoverRegistry;
    xoverRegistry.registerFactory("my_xover",
        []() -> std::unique_ptr<BaseCrossover> {
            return std::make_unique<MyXover>();
        }
    );

    auto op = xoverRegistry.create("my_xover");
    std::cout << "Loaded plugin: my_xover\n";
    return 0;
}
```

### Python

> **Not available in Python bindings yet.**
> The plugin registry is implemented in `include/ga/plugin/registry.hpp`
> (C++ only).

---

## 25. Benchmark Suite

Test operator throughput and convergence quality on standard functions.

### C++

```cpp
#include "benchmark/ga_benchmark.h"

int main() {
    GABenchmark suite;
    suite.runAllBenchmarks();   // runs operators + functions + scalability
    suite.generateReport();
    suite.exportToCSV("benchmark_results.csv");
    return 0;
}
```

Or via the pre-built executable:

```bash
# Build benchmark target
cmake --build build

# Run all benchmarks
./build/bin/ga-benchmark --all

# Operator performance only
./build/bin/ga-benchmark --operators

# Function convergence only
./build/bin/ga-benchmark --functions

# Scalability sweep
./build/bin/ga-benchmark --scalability

# Export CSV
./build/bin/ga-benchmark --all --csv
```

### Python

The benchmark suite is exposed in Python through `ga.BenchmarkConfig` and
`ga.GABenchmark`:

```python
import ga

cfg = ga.BenchmarkConfig()
cfg.warmup_iterations = 1
cfg.benchmark_iterations = 3
cfg.verbose = False

b = ga.GABenchmark(cfg)
b.run_operator_benchmarks()
print("Operator rows:", len(b.operator_results()))

b.run_function_benchmarks()
print("Function rows:", len(b.function_results()))

b.export_to_csv("benchmark_results.csv")
```

---

## 26. C API

A C-compatible interface for embedding the framework in C or FFI consumers.

### C

```c
#include <ga/c_api.h>
#include <stdio.h>

static double sphere_fitness(const double* genes, int length, void* user_data) {
    (void)user_data;
    double sum = 0.0;
    for (int i = 0; i < length; ++i) sum += genes[i] * genes[i];
    return 1000.0 / (1.0 + sum);
}

int main(void) {
    ga_config_c cfg = {
        .population_size = 60,
        .generations     = 100,
        .dimension       = 10,
        .crossover_rate  = 0.8,
        .mutation_rate   = 0.05,
        .bounds_lower    = -5.12,
        .bounds_upper    =  5.12,
        .elite_ratio     = 0.05,
        .seed            = 42
    };

    if (ga_validate_config(&cfg) != GA_STATUS_OK) return 1;

    ga_handle* h = ga_create(&cfg);
    if (!h) return 1;

    if (ga_run(h, sphere_fitness, NULL) != GA_STATUS_OK) {
        ga_destroy(h); return 1;
    }

    printf("Best fitness: %f\n", ga_best_fitness(h));

    int n = ga_history_length(h);
    double best_hist[1024], avg_hist[1024];
    if (n > 0 && n <= 1024) {
        ga_best_history(h, best_hist, n);
        ga_avg_history(h,  avg_hist,  n);
        printf("Gen 0 best: %f\n", best_hist[0]);
    }

    ga_destroy(h);
    return 0;
}
```

### Python

> The C API is a **C-only interface** and is not exposed to Python.
> Use the native Python bindings (`import ga`) instead.

---

## 27. Reproducibility Controls

Seed-based deterministic runs.

### C++

```cpp
#include <ga/genetic_algorithm.hpp>

ga::Config cfg;
cfg.seed = 42;   // fixed seed → reproducible results
// cfg.seed = 0; // 0 → random seed each run

ga::GeneticAlgorithm alg(cfg);
ga::Result r1 = alg.run(my_fitness);

// Re-run with same seed → identical result
ga::GeneticAlgorithm alg2(cfg);
ga::Result r2 = alg2.run(my_fitness);

assert(r1.bestFitness == r2.bestFitness);
```

### Python

```python
import ga

def my_fitness(x):
    return 1000.0 / (1.0 + sum(xi**2 for xi in x))

cfg = ga.Config()
cfg.population_size = 50
cfg.generations     = 100
cfg.dimension       = 5
cfg.bounds          = ga.Bounds(-5.12, 5.12)
cfg.seed            = 42   # fixed → reproducible

r1 = ga.GeneticAlgorithm(cfg).run(my_fitness)
r2 = ga.GeneticAlgorithm(cfg).run(my_fitness)

assert r1.best_fitness == r2.best_fitness, "Runs should be identical"
print(f"Reproducible best: {r1.best_fitness:.6f}")
```

> **Checkpointing RNG state:** The `ga.CheckpointState.rng_state` field holds
> a string token for future RNG restore support.  Full deterministic
> mid-run resume is implemented in C++ via
> `ga::checkpoint::CheckpointManager::loadJson`.

---

## Building the Python Module

```bash
# 1. Install pybind11
pip install pybind11

# 2. Configure and build
mkdir -p build && cd build
cmake ..
cmake --build . --target ga_python_module -j$(nproc)

# 3. Add the build directory to PYTHONPATH
export PYTHONPATH="$(pwd)/python:$PYTHONPATH"

# 4. Test import
python3 -c "import ga; print(ga.__doc__)"
```

Run the bundled Python example:

```bash
python3 python/example.py
```

---

## Summary of Python Bindings Coverage

| Module symbol | Description |
|---------------|-------------|
| `ga.Config` | Algorithm configuration |
| `ga.Bounds` | Gene search bounds |
| `ga.Result` | Run results (best genes, fitness, history) |
| `ga.OptimizationResult` | Generic optimization result container |
| `ga.Evaluation` | Evaluation / objective record |
| `ga.Individual` | Generic individual container |
| `ga.IGenome` | Genome interface base class |
| `ga.GeneticAlgorithm` | Main single-objective GA engine |
| `ga.GeneticAlgorithm.set_crossover_operator` | Plug in a crossover operator |
| `ga.GeneticAlgorithm.set_mutation_operator` | Plug in a mutation operator |
| `ga.make_one_point_crossover` | Factory: one-point crossover |
| `ga.make_two_point_crossover` | Factory: two-point crossover |
| `ga.make_gaussian_mutation` | Factory: Gaussian mutation |
| `ga.make_uniform_mutation` | Factory: Uniform mutation |
| **Evaluation** | |
| `ga.ParallelEvaluator` | Threaded batch evaluator over candidate vectors |
| `ga.LocalDistributedExecutor` | Threaded distributed executor over candidate batches |
| **Selection Helpers** | |
| `ga.selection_tournament_indices` | Tournament selection over fitness list |
| `ga.selection_roulette_indices` | Roulette-wheel selection over fitness list |
| `ga.selection_rank_indices` | Rank-based selection over fitness list |
| `ga.selection_sus_indices` | Stochastic universal sampling over fitness list |
| `ga.selection_elitism_indices` | Elitism/top-k selection over fitness list |
| **Benchmark** | |
| `ga.BenchmarkConfig` | Configure benchmark warmup/iterations/output |
| `ga.BenchmarkResult` | Scalability benchmark summary row |
| `ga.OperatorBenchmark` | Operator benchmark row |
| `ga.FunctionBenchmark` | Function optimization benchmark row |
| `ga.GABenchmark` | Run benchmark suite and export reports/CSV |
| **Representations** | |
| `ga.VectorGenome` | Real-valued genome (`double`) |
| `ga.BitsetGenome` | Binary/bitset genome |
| `ga.PermutationGenome` | Permutation genome with ordering utilities |
| `ga.SetGenome` | Set-based genome |
| `ga.MapGenome` | Map/dictionary genome |
| `ga.NdArrayGenome` | N-dimensional array genome |
| `ga.TreeGenome` | Tree genome for genetic programming |
| **Genetic Programming** | |
| `ga.ValueType` | Enum of GP value types (`any`, `bool`, `int`, `double`) |
| `ga.Signature` | GP primitive signature (return type + arg types) |
| `ga.Primitive` | GP primitive descriptor |
| `ga.Node` | GP tree node (symbol, return type, children) |
| `ga.TreeBuilder` | Random GP tree generator (grow method) |
| `ga.ADFPool` | Automatically Defined Function registry |
| **Multi-Objective: NSGA-II** | |
| `ga.Nsga2Config` | NSGA-II configuration |
| `ga.Nsga2` | NSGA-II objective-space utilities |
| `ga.nsga2_non_dominated_sort` | Convenience: non-dominated sorting |
| `ga.nsga2_crowding_distance` | Convenience: crowding distance |
| **Multi-Objective: NSGA-III** | |
| `ga.Nsga3` | NSGA-III objective-space utilities |
| `ga.nsga3_reference_points` | Convenience: Das-Dennis reference points |
| `ga.nsga3_environmental_select_indices` | Convenience: NSGA-III environmental selection |
| **Multi-Objective: SPEA2** | |
| `ga.Spea2` | SPEA2 objective-space utilities |
| `ga.spea2_strength_fitness` | Convenience: SPEA2 strength fitness values |
| `ga.spea2_environmental_select_indices` | Convenience: SPEA2 environmental selection indices |
| **Multi-Objective: MO-CMA-ES** | |
| `ga.MoCmaEsConfig` | MO-CMA-ES configuration |
| `ga.MoCmaEsResult` | MO-CMA-ES result container |
| `ga.MoCmaEs` | Multi-objective CMA-ES wrapper |
| **Evolution Strategies** | |
| `ga.EvolutionStrategyConfig` | ES configuration (μ, λ, σ, etc.) |
| `ga.EvolutionStrategyResult` | ES result container |
| `ga.EvolutionStrategy` | (μ,λ)/(μ+λ) evolution strategy |
| **CMA-ES** | |
| `ga.CmaEsConfig` | CMA-ES configuration |
| `ga.CmaEsResult` | CMA-ES result container |
| `ga.DiagonalCmaEs` | Diagonal CMA-ES optimizer |
| **High-Level Optimizer API** | |
| `ga.Optimizer` | Fluent high-level optimizer facade |
| `ga.OptimizerBuilder` | Builder for `ga.Optimizer` |
| `ga.MultiObjectiveResult` | Multi-objective optimizer result |
| **Constraint Handling** | |
| `ga.ConstraintSet` | Hard constraints, soft penalties, and repairs |
| `ga.is_feasible` | Check all hard constraints |
| `ga.total_penalty` | Sum all soft penalties |
| `ga.apply_repairs` | Apply all repair functions to a gene vector |
| `ga.penalized_fitness` | Base fitness adjusted for constraint violations |
| **Adaptive Operators** | |
| `ga.AdaptiveRates` | Mutation/crossover rate container |
| `ga.AdaptiveRateController` | Dynamic rate controller based on diversity/progress |
| **Hybrid Optimization** | |
| `ga.HybridOptimizer` | GA with optional local search (memetic) |
| **Co-Evolution** | |
| `ga.CoevolutionConfig` | Co-evolution configuration |
| `ga.CoevolutionEngine` | Multi-population co-evolution engine |
| **Checkpointing** | |
| `ga.CheckpointState` | Checkpoint data container |
| `ga.checkpoint_save_json` | Save checkpoint to JSON |
| `ga.checkpoint_load_json` | Load checkpoint from JSON |
| `ga.checkpoint_save_binary` | Save checkpoint to binary file |
| `ga.checkpoint_load_binary` | Load checkpoint from binary file |
| **Experiment Tracking** | |
| `ga.ExperimentTracker` | Logs config, history CSV, and best-solution CSV |
| **Visualization / CSV Export** | |
| `ga.export_fitness_curve_csv` | Write fitness curve (best + avg) to CSV |
| `ga.export_pareto_front_csv` | Write Pareto front objectives to CSV |
| `ga.export_diversity_csv` | Write per-generation diversity values to CSV |

---

## See Also

- [README.md](README.md) — Build instructions and project overview
- [FEATURE_CHECKLIST.md](FEATURE_CHECKLIST.md) — Complete feature roadmap
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) — Implementation details and test status
- [ARCHITECTURE.md](ARCHITECTURE.md) — Design overview
- [python/example.py](python/example.py) — Runnable Python demo
