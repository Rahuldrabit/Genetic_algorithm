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
| 3 | [Chromosome Representations](#3-chromosome-representations) | ✅ | ⚠️ real-valued only |
| 4 | [Crossover Operators](#4-crossover-operators) | ✅ | ⚠️ 2 operators exposed |
| 5 | [Mutation Operators](#5-mutation-operators) | ✅ | ⚠️ 2 operators exposed |
| 6 | [Selection Operators](#6-selection-operators) | ✅ | ❌ not exposed |
| 7 | [Core GA Run and Results](#7-core-ga-run-and-results) | ✅ | ✅ |
| 8 | [High-Level Optimizer API](#8-high-level-optimizer-api) | ✅ | ❌ not exposed |
| 9 | [Multi-Objective: NSGA-II](#9-multi-objective-nsga-ii) | ✅ | ✅ (objective-space utils) |
| 10 | [Multi-Objective: NSGA-III](#10-multi-objective-nsga-iii) | ✅ | ✅ |
| 11 | [Multi-Objective: SPEA2](#11-multi-objective-spea2) | ✅ | ❌ not exposed |
| 12 | [Multi-Objective: MO-CMA-ES](#12-multi-objective-mo-cma-es) | ✅ | ❌ not exposed |
| 13 | [Evolution Strategies (ES)](#13-evolution-strategies-es) | ✅ | ❌ not exposed |
| 14 | [CMA-ES](#14-cma-es) | ✅ | ❌ not exposed |
| 15 | [Genetic Programming (GP)](#15-genetic-programming-gp) | ✅ | ❌ not exposed |
| 16 | [Adaptive Operators](#16-adaptive-operators) | ✅ | ❌ not exposed |
| 17 | [Hybrid Optimization](#17-hybrid-optimization) | ✅ | ❌ not exposed |
| 18 | [Constraint Handling](#18-constraint-handling) | ✅ | ❌ not exposed |
| 19 | [Parallel and Distributed Evaluation](#19-parallel-and-distributed-evaluation) | ✅ | ❌ not exposed |
| 20 | [Co-Evolution](#20-co-evolution) | ✅ | ❌ not exposed |
| 21 | [Checkpointing](#21-checkpointing) | ✅ | ✅ |
| 22 | [Experiment Tracking](#22-experiment-tracking) | ✅ | ❌ not exposed |
| 23 | [Visualization and CSV Export](#23-visualization-and-csv-export) | ✅ | ❌ not exposed |
| 24 | [Plugin Architecture](#24-plugin-architecture) | ✅ | ❌ not exposed |
| 25 | [Benchmark Suite](#25-benchmark-suite) | ✅ | ❌ not exposed |
| 26 | [C API](#26-c-api) | ✅ | N/A (C only) |
| 27 | [Reproducibility Controls](#27-reproducibility-controls) | ✅ | ✅ |

**Legend:** ✅ fully available · ⚠️ partially available · ❌ not yet exposed in Python bindings

---

## 1. Quick Start

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
print("Best genes:",   result.best_genes)
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

The library provides six genome types for different problem domains.

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

The default `ga.GeneticAlgorithm` operates on real-valued gene vectors.
No explicit genome construction is needed — the engine creates them internally.

```python
import ga

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

> **Not available in Python bindings yet.**
> The Python `ga.GeneticAlgorithm` class operates exclusively on real-valued
> (`double`) gene vectors. Integer-specific crossover/mutation is not currently
> exposed. As a workaround, round real-valued genes to integers in your fitness
> function:
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

> **Not available in Python bindings yet.**
> The Python engine is real-valued. Use integer rounding as a workaround (see §3.2).

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

> **Not available in Python bindings yet.**
> Permutation-specific operators (Order Crossover, PMX, Cycle Crossover,
> Scramble Mutation, etc.) are not exposed. There is no direct Python equivalent
> for the interactive permutation mode.

### 3.5 Set Genome

#### C++

```cpp
#include <ga/representations/set_genome.hpp>

ga::SetGenome<int> sg({1, 3, 5, 7});
std::cout << sg.encodingName() << "\n"; // "set<int>"
```

#### Python

> **Not available in Python bindings yet.**

### 3.6 Map / Dictionary Genome

#### C++

```cpp
#include <ga/representations/map_genome.hpp>

ga::MapGenome<std::string, double> mg({{"x", 1.0}, {"y", -0.5}});
std::cout << mg.encodingName() << "\n"; // "map<string, double>"
```

#### Python

> **Not available in Python bindings yet.**

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

> **Not available in Python bindings yet.**

---

## 4. Crossover Operators

### Available Operators

| Operator | Class | Suitable For |
|----------|-------|--------------|
| One-Point | `OnePointCrossover` | binary, real, integer |
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

Only **one-point** and **two-point** crossover are exposed as factory functions.

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
> crossovers are **not available in Python bindings yet**.

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

Only **Gaussian** and **Uniform** mutation are exposed.

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
> **not available in Python bindings yet**.

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

> **Not available in Python bindings yet.**
> Selection operators are not individually exposed to Python.
> The `ga.GeneticAlgorithm` uses an internal tournament-style selection
> that cannot be swapped from Python currently.

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

> **Not available in Python bindings yet.**
> The high-level `ga::api::Optimizer` C++ class is not bound to Python.
> Use `ga.GeneticAlgorithm` directly (see [§7](#7-core-ga-run-and-results)).

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

### Python

> **Not available in Python bindings yet.**
> SPEA2 is implemented in `include/ga/moea/spea2.hpp` (C++ only).

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

> **Not available in Python bindings yet.**
> MO-CMA-ES is implemented in `include/ga/moea/mo_cmaes.hpp` (C++ only).

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

> **Not available in Python bindings yet.**
> Evolution Strategies are implemented in `include/ga/es/evolution_strategies.hpp`
> (C++ only).

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

> **Not available in Python bindings yet.**
> CMA-ES is implemented in `include/ga/es/cmaes.hpp` (C++ only).

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

### 15.2 Subtree crossover (C++)

```cpp
#include "crossover/subtree_crossover.h"

// Build parent trees (TreeNode* from base_crossover.h)
TreeNode* parent1 = /* ... build tree ... */;
TreeNode* parent2 = /* ... build tree ... */;

SubtreeCrossover xover(/*seed=*/42);
auto [child1, child2] = xover.crossover(parent1, parent2);
```

### 15.3 Python

> **Not available in Python bindings yet.**
> The GP node system, tree builder, ADF pool, and subtree crossover are
> implemented in `include/ga/gp/` and `crossover/subtree_crossover.*`
> (C++ only).

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

> **Not available in Python bindings yet.**
> `ga::adaptive::AdaptiveRateController` is implemented in
> `include/ga/adaptive/adaptive_policy.hpp` (C++ only).

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

> **Not available in Python bindings yet.**
> `ga::hybrid::HybridOptimizer` is implemented in
> `include/ga/hybrid/hybrid_optimizer.hpp` (C++ only).

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

> **Not available in Python bindings yet.**
> Constraint handling is implemented in `include/ga/constraints/constraints.hpp`
> (C++ only).

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

> **Not available in Python bindings yet.**
> Parallel and distributed evaluators are implemented in
> `include/ga/evaluation/` (C++ only).
> As a workaround, Python's `concurrent.futures` can parallelize fitness calls
> externally and pass results to a Python-level custom fitness function.

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

> **Not available in Python bindings yet.**
> Co-evolution is implemented in `include/ga/coevolution/coevolution.hpp`
> (C++ only).

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

# Reload
loaded = ga.checkpoint_load_json("run.json")
print("Generation:", loaded.generation)
print("Best fitness:", loaded.result.best_fitness)
```

> **Note:** Binary checkpoint (`saveBinary` / `loadBinary`) is C++ only;
> Python bindings expose the JSON checkpoint API only.

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

> **Not available in Python bindings yet.**
> `ga::tracking::ExperimentTracker` is implemented in
> `include/ga/tracking/experiment_tracker.hpp` (C++ only).
>
> As a workaround in Python, log metrics manually:
>
> ```python
> import ga, csv
>
> result = engine.run(my_fitness)
>
> with open("metrics.csv", "w", newline="") as f:
>     w = csv.writer(f)
>     w.writerow(["generation", "best_fitness", "avg_fitness"])
>     for gen, (b, a) in enumerate(zip(result.best_history, result.avg_history)):
>         w.writerow([gen, b, a])
> ```

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

> **Not available in Python bindings yet.**
> Visualization/export helpers are in `include/ga/visualization/export.hpp`
> (C++ only).
>
> As a workaround in Python, export from `result.best_history` directly:
>
> ```python
> import ga, csv
>
> result = engine.run(my_fitness)
>
> with open("fitness.csv", "w", newline="") as f:
>     w = csv.writer(f)
>     w.writerow(["generation", "best_fitness", "avg_fitness"])
>     for gen, (b, a) in enumerate(zip(result.best_history, result.avg_history)):
>         w.writerow([gen, b, a])
> print("Exported fitness.csv")
> ```

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

> **Not available in Python bindings yet.**
> The benchmark suite is implemented in `benchmark/` and exposed via the
> `ga-benchmark` executable (C++ only).
>
> You can replicate benchmark-style measurements in Python using the
> `ga.GeneticAlgorithm` directly:
>
> ```python
> import ga, time
>
> def sphere(x):
>     return 1000.0 / (1.0 + sum(xi**2 for xi in x))
>
> for dim in [5, 10, 20]:
>     cfg = ga.Config()
>     cfg.population_size = 60
>     cfg.generations     = 100
>     cfg.dimension       = dim
>     cfg.bounds          = ga.Bounds(-5.12, 5.12)
>     t0 = time.perf_counter()
>     r  = ga.GeneticAlgorithm(cfg).run(sphere)
>     elapsed = time.perf_counter() - t0
>     print(f"dim={dim:2d}  best={r.best_fitness:.4f}  time={elapsed*1000:.1f}ms")
> ```

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
cmake --build . --target ga-python-bindings -j$(nproc)

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
| `ga.GeneticAlgorithm` | Main single-objective GA engine |
| `ga.GeneticAlgorithm.set_crossover_operator` | Plug in a crossover operator |
| `ga.GeneticAlgorithm.set_mutation_operator` | Plug in a mutation operator |
| `ga.make_one_point_crossover` | Factory: one-point crossover |
| `ga.make_two_point_crossover` | Factory: two-point crossover |
| `ga.make_gaussian_mutation` | Factory: Gaussian mutation |
| `ga.make_uniform_mutation` | Factory: Uniform mutation |
| `ga.Nsga2Config` | NSGA-II configuration |
| `ga.Nsga2` | NSGA-II objective-space utilities |
| `ga.nsga2_non_dominated_sort` | Convenience: non-dominated sorting |
| `ga.nsga2_crowding_distance` | Convenience: crowding distance |
| `ga.Nsga3` | NSGA-III objective-space utilities |
| `ga.nsga3_reference_points` | Convenience: Das-Dennis reference points |
| `ga.nsga3_environmental_select_indices` | Convenience: NSGA-III selection |
| `ga.CheckpointState` | Checkpoint data container |
| `ga.checkpoint_save_json` | Save checkpoint to JSON |
| `ga.checkpoint_load_json` | Load checkpoint from JSON |

---

## See Also

- [README.md](README.md) — Build instructions and project overview
- [FEATURE_CHECKLIST.md](FEATURE_CHECKLIST.md) — Complete feature roadmap
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) — Implementation details and test status
- [ARCHITECTURE.md](ARCHITECTURE.md) — Design overview
- [python/example.py](python/example.py) — Runnable Python demo
