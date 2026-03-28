# User Guide – Genetic Algorithm Framework

This guide explains every feature available in this framework and shows you how to use each one. Each section includes a short description and a minimal example so you can get started immediately.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Configuration](#2-configuration)
3. [Chromosome Representations](#3-chromosome-representations)
4. [Crossover Operators](#4-crossover-operators)
5. [Mutation Operators](#5-mutation-operators)
6. [Selection Operators](#6-selection-operators)
7. [Benchmark Functions](#7-benchmark-functions)
8. [C++ API](#8-c-api)
9. [High-Level Optimizer API](#9-high-level-optimizer-api)
10. [Python API](#10-python-api)
11. [C API](#11-c-api)
12. [Multi-Objective Optimization (NSGA-II / NSGA-III)](#12-multi-objective-optimization-nsga-ii--nsga-iii)
13. [Evolution Strategies (ES / CMA-ES)](#13-evolution-strategies-es--cma-es)
14. [Genetic Programming (GP)](#14-genetic-programming-gp)
15. [Parallel & Distributed Evaluation](#15-parallel--distributed-evaluation)
16. [Checkpointing](#16-checkpointing)
17. [Adaptive Operators](#17-adaptive-operators)
18. [Hybrid Optimization](#18-hybrid-optimization)
19. [Constraint Handling](#19-constraint-handling)
20. [Experiment Tracking](#20-experiment-tracking)
21. [Visualization & Export](#21-visualization--export)
22. [Plugin Architecture](#22-plugin-architecture)
23. [Benchmark Suite](#23-benchmark-suite)
24. [Building the Project](#24-building-the-project)
25. [Running the Interactive Demo](#25-running-the-interactive-demo)
26. [Troubleshooting](#26-troubleshooting)

---

## 1. Quick Start

The fastest way to run your first optimization:

```cpp
#include <ga/genetic_algorithm.hpp>
#include <cmath>
#include <iostream>

// Your fitness function (higher value = better solution)
double sphere(const std::vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) sum += xi * xi;
    return 1000.0 / (1.0 + sum);   // convert minimization -> maximization
}

int main() {
    ga::Config cfg;
    cfg.populationSize = 50;
    cfg.generations    = 100;
    cfg.dimension      = 10;
    cfg.bounds         = {-5.12, 5.12};

    ga::GeneticAlgorithm ga(cfg);
    ga::Result result = ga.run(sphere);

    std::cout << "Best fitness: " << result.bestFitness << "\n";
}
```

Build and run:

```bash
mkdir build && cd build
cmake ..
cmake --build .
./examples/ga-minimal
```

---

## 2. Configuration

All tuning parameters live in `ga::Config` (`include/ga/config.hpp`).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `populationSize` | `50` | Number of individuals per generation. Larger → more diversity but slower. |
| `generations` | `100` | How many evolution cycles to run. |
| `dimension` | `10` | Number of genes (variables) per individual. |
| `crossoverRate` | `0.8` | Probability that two parents produce offspring via crossover. |
| `mutationRate` | `0.1` | Probability that a gene is mutated. |
| `bounds.lower` | `-5.12` | Minimum gene value for real-valued individuals. |
| `bounds.upper` | `5.12` | Maximum gene value. |
| `eliteRatio` | `0.05` | Fraction of best individuals copied unchanged to the next generation (elitism). |
| `seed` | `0` | Random seed. `0` = use `std::random_device` (non-deterministic). Set a fixed value for reproducible results. |

```cpp
ga::Config cfg;
cfg.populationSize = 100;
cfg.generations    = 200;
cfg.dimension      = 20;
cfg.crossoverRate  = 0.9;
cfg.mutationRate   = 0.05;
cfg.bounds         = {-10.0, 10.0};
cfg.eliteRatio     = 0.10;
cfg.seed           = 42;   // reproducible
```

---

## 3. Chromosome Representations

The framework supports four chromosome types across two levels of API:

- **`ga::GeneticAlgorithm` (public API)** — always uses `std::vector<double>` (real-valued genes). The crossover and mutation operators you set via `setCrossoverOperator` / `setMutationOperator` must be compatible with `double` vectors.
- **Legacy operator modules** (`crossover/`, `mutation/`, `selection-operator/`) — contain operator classes that also work with binary (`BitString`) and permutation (`std::vector<int>`) chromosomes. These are used directly when you build your own evolutionary loop (e.g., NSGA-II, GP, or the interactive demo).

### 3.1 Binary (`BitString` = `std::vector<bool>`)

A vector of `bool` values (`0` or `1`).  
**Best for:** feature selection, binary optimization, combinatorial problems where choices are yes/no.

Compatible operators: One-point, Two-point, Uniform crossover; Bit-flip mutation.

### 3.2 Real-Valued (`std::vector<double>`)

A vector of `double` values within configurable bounds. **This is the type used by `ga::GeneticAlgorithm`.**  
**Best for:** continuous function optimization, parameter tuning, neural network weights.

Compatible operators: Arithmetic, Blend (BLX-α), SBX, One-point, Two-point, Uniform crossover; Gaussian, Uniform mutation.

### 3.3 Integer (`std::vector<int>`)

A vector of integer values.  
**Best for:** discrete scheduling, resource allocation, index-based problems.

Compatible operators: One-point, Two-point, Uniform, Arithmetic crossover; Random resetting, Creep mutation.

### 3.4 Permutation (`std::vector<int>` with unique values)

A vector of integers where each value appears exactly once — an ordering.  
**Best for:** Traveling Salesman Problem (TSP), job scheduling, route optimization.

Compatible operators: Order (OX), Partially Mapped (PMX), Cycle (CX) crossover; Swap, Insert, Scramble, Inversion mutation.

---

## 4. Crossover Operators

Crossover combines two parent chromosomes to produce offspring. Select the operator that suits your representation.

> **Factory vs. direct construction:** The public `ga::` namespace only has `makeOnePointCrossover()` and `makeTwoPointCrossover()` as convenience factories (declared in `include/ga/genetic_algorithm.hpp`). All other operators must be constructed directly by including their header and using `std::make_unique<ClassName>(...)`.

### 4.1 One-Point Crossover

Splits both parents at a single random point and swaps their tails.  
**Use with:** Binary, Real-valued, Integer.

```cpp
// Public factory (include/ga/genetic_algorithm.hpp)
alg.setCrossoverOperator(ga::makeOnePointCrossover());
```

### 4.2 Two-Point Crossover

Two random cut points divide the chromosome into three segments; the middle segment is swapped.  
**Use with:** Binary, Real-valued, Integer.

```cpp
// Public factory (include/ga/genetic_algorithm.hpp)
alg.setCrossoverOperator(ga::makeTwoPointCrossover());
```

### 4.3 Uniform Crossover

Each gene is independently inherited from either parent with equal probability.  
**Use with:** Binary, Real-valued, Integer.  
**Tip:** Produces more diversity than point-based crossovers.

```cpp
#include "crossover/uniform_crossover.h"
alg.setCrossoverOperator(std::make_unique<UniformCrossover>(/*probability=*/0.5));
```

### 4.4 Arithmetic Crossover

Each child gene is a weighted average of the two parent genes: `α·p1 + (1−α)·p2`.  
**Use with:** Real-valued, Integer.

```cpp
#include "crossover/crossover_all.h"   // or the specific header
alg.setCrossoverOperator(std::make_unique<ArithmeticCrossover>());
```

### 4.5 Blend Crossover (BLX-α)

Extends the search range slightly beyond the parents, exploring values around them.  
**Use with:** Real-valued.  
**Tip:** α = 0.5 is a common default, giving good exploration.

```cpp
#include "crossover/blend_crossover.h"
alg.setCrossoverOperator(std::make_unique<BlendCrossover>(/*alpha=*/0.5));
```

### 4.6 Simulated Binary Crossover (SBX)

Mimics the behavior of one-point crossover for binary strings in real-valued space.  
**Use with:** Real-valued.  
**Tip:** Widely used in NSGA-II and other evolutionary algorithms.

```cpp
#include "crossover/simulated_binary_crossover.h"
alg.setCrossoverOperator(std::make_unique<SimulatedBinaryCrossover>(/*eta=*/2.0));
```

### 4.7 Order Crossover (OX)

Preserves the relative order of elements from one parent while filling the rest from the other.  
**Use with:** Permutation.

```cpp
#include "crossover/order_crossover.h"
alg.setCrossoverOperator(std::make_unique<OrderCrossover>());
```

### 4.8 Partially Mapped Crossover (PMX)

Preserves absolute position information by creating a partial mapping between the parents.  
**Use with:** Permutation.

```cpp
#include "crossover/partially_mapped_crossover.h"
alg.setCrossoverOperator(std::make_unique<PartiallyMappedCrossover>());
```

### 4.9 Cycle Crossover (CX)

Identifies cycles of corresponding positions between parents and alternates which parent supplies each cycle.  
**Use with:** Permutation.

```cpp
#include "crossover/cycle_crossover.h"
alg.setCrossoverOperator(std::make_unique<CycleCrossover>());
```

### 4.10 Differential Evolution Crossover

Perturbs a base individual using the difference of two others. Often used with the DE/rand/1 strategy.  
**Use with:** Real-valued.

```cpp
#include "crossover/differential_evolution_crossover.h"
alg.setCrossoverOperator(std::make_unique<DifferentialEvolutionCrossover>(/*F=*/0.8, /*CR=*/0.9));
```

### 4.11 Multi-Point Crossover

Generalization of one- and two-point crossover: uses *k* random cut points.  
**Use with:** Binary, Real-valued, Integer.

```cpp
#include "crossover/multi_point_crossover.h"
alg.setCrossoverOperator(std::make_unique<MultiPointCrossover>(/*k=*/3));
```

### 4.12 Uniform K-Vector Crossover

Applies uniform crossover independently to each gene dimension using a separate probability mask.  
**Use with:** Real-valued.

```cpp
#include "crossover/uniform_k_vector_crossover.h"
alg.setCrossoverOperator(std::make_unique<UniformKVectorCrossover>());
```

### 4.13 Edge Assembly Crossover (EAX)

Preserves edges (adjacency pairs) from both parents.  
**Use with:** Permutation (graph/TSP problems).

```cpp
#include "crossover/edge_crossover.h"
alg.setCrossoverOperator(std::make_unique<EdgeCrossover>());
```

### 4.14 Cut-and-Crossfill Crossover

Cuts at a random point; the remaining genes are filled from the other parent in their original order.  
**Use with:** Permutation.

```cpp
#include "crossover/cut_and_crossfill_crossover.h"
alg.setCrossoverOperator(std::make_unique<CutAndCrossfillCrossover>());
```

### 4.15 Line Recombination

Offspring are placed on the line connecting the two parents in gene space.  
**Use with:** Real-valued.

```cpp
#include "crossover/line_recombination.h"
alg.setCrossoverOperator(std::make_unique<LineRecombination>());
```

### 4.16 Intermediate Recombination

Each offspring gene is a random blend between the corresponding parent genes.  
**Use with:** Real-valued.

```cpp
#include "crossover/intermediate_recombination.h"
alg.setCrossoverOperator(std::make_unique<IntermediateRecombination>());
```

### 4.17 Diploid Recombination

Maintains a dominant and recessive copy of each gene; the expressed value follows dominance rules.  
**Use with:** Binary.

```cpp
#include "crossover/diploid_recombination.h"
alg.setCrossoverOperator(std::make_unique<DiploidRecombination>());
```

### 4.18 Subtree Crossover

Swaps randomly selected subtrees between two tree-based individuals.  
**Use with:** Genetic Programming (tree representation).

```cpp
#include "crossover/subtree_crossover.h"
alg.setCrossoverOperator(std::make_unique<SubtreeCrossover>());
```

---

## 5. Mutation Operators

Mutation randomly modifies one or more genes to maintain diversity and avoid premature convergence.

> **Factory vs. direct construction:** The public `ga::` namespace only provides `makeGaussianMutation()` and `makeUniformMutation()` as convenience factories. All other mutation operators must be constructed directly by including their header.

### 5.1 Bit-Flip Mutation

Flips each bit independently with probability `mutationRate`.  
**Use with:** Binary.

```cpp
#include "mutation/bit_flip_mutation.h"
alg.setMutationOperator(std::make_unique<BitFlipMutation>());
```

### 5.2 Gaussian Mutation

Adds a small Gaussian random noise to each gene. Keeps the search local.  
**Use with:** Real-valued.

```cpp
// Public factory (include/ga/genetic_algorithm.hpp)
alg.setMutationOperator(ga::makeGaussianMutation());
```

### 5.3 Uniform Mutation

Replaces a gene with a uniformly random value within bounds.  
**Use with:** Real-valued.

```cpp
// Public factory (include/ga/genetic_algorithm.hpp)
alg.setMutationOperator(ga::makeUniformMutation());
```

### 5.4 Swap Mutation

Picks two random positions and swaps their values.  
**Use with:** Permutation.

```cpp
#include "mutation/swap_mutation.h"
alg.setMutationOperator(std::make_unique<SwapMutation>());
```

### 5.5 Insert Mutation

Removes a gene from a random position and inserts it at another random position.  
**Use with:** Permutation.

```cpp
#include "mutation/insert_mutation.h"
alg.setMutationOperator(std::make_unique<InsertMutation>());
```

### 5.6 Scramble Mutation

Selects a random subset of genes and shuffles them in place.  
**Use with:** Permutation.

```cpp
#include "mutation/scramble_mutation.h"
alg.setMutationOperator(std::make_unique<ScrambleMutation>());
```

### 5.7 Inversion Mutation

Reverses the order of genes between two random positions.  
**Use with:** Permutation.

```cpp
#include "mutation/inversion_mutation.h"
alg.setMutationOperator(std::make_unique<InversionMutation>());
```

### 5.8 Creep Mutation

Adds or subtracts a small constant (creep) to an integer gene.  
**Use with:** Integer.

```cpp
#include "mutation/creep_mutation.h"
alg.setMutationOperator(std::make_unique<CreepMutation>(/*step=*/1));
```

### 5.9 Random Resetting Mutation

Replaces a gene with a randomly chosen integer from a valid range.  
**Use with:** Integer.

```cpp
#include "mutation/random_resetting_mutation.h"
alg.setMutationOperator(std::make_unique<RandomResettingMutation>());
```

### 5.10 Self-Adaptive Mutation

Each individual carries its own mutation step-size, which also evolves alongside the genes. This is the foundation of Evolution Strategies.  
**Use with:** Real-valued (ES contexts).

```cpp
#include "mutation/self_adaptive_mutation.h"
alg.setMutationOperator(std::make_unique<SelfAdaptiveMutation>());
```

### 5.11 List Mutation

General mutation for list-based genomes (insert, delete, or replace list elements).  
**Use with:** Variable-length representations.

```cpp
#include "mutation/list_mutation.h"
alg.setMutationOperator(std::make_unique<ListMutation>());
```

---

## 6. Selection Operators

Selection decides which individuals survive or reproduce based on their fitness.

> **Note:** `ga::GeneticAlgorithm` (the public C++ API) uses tournament selection internally and does not expose a `setSelectionOperator` hook. The selection classes in `selection-operator/` are designed for use in custom evolutionary loops (e.g., the interactive demo, NSGA-II, or your own main loop). In the examples below, `sel` refers to a selection operator you instantiate directly.

### 6.1 Tournament Selection

Randomly samples *k* individuals and picks the best among them. A common default.  
**Use with:** All representations.  
**Tip:** Higher tournament size (*k*) → stronger selection pressure.

```cpp
#include "selection-operator/tournament_selection.h"
TournamentSelection sel(/*k=*/3);
// int winner = sel.select(population, fitnesses);
```

### 6.2 Roulette Wheel Selection (Fitness-Proportionate)

Each individual's probability of selection is proportional to its fitness.  
**Use with:** All representations.  
**Caution:** Sensitive to fitness scaling; a dominant individual can take over quickly.

```cpp
#include "selection-operator/roulette_wheel_selection.h"
RouletteWheelSelection sel;
```

### 6.3 Rank Selection

Individuals are ranked by fitness; selection probability is proportional to rank, not raw fitness. More stable than roulette wheel.  
**Use with:** All representations.

```cpp
#include "selection-operator/rank_selection.h"
RankSelection sel;
```

### 6.4 Elitism Selection

The top fraction of individuals (controlled by `eliteRatio`) is copied unchanged into the next generation.  
**Use with:** All representations (built into the GA engine automatically).

```cpp
// Configured via Config; no separate operator needed for ga::GeneticAlgorithm
cfg.eliteRatio = 0.10;   // top 10% survive unchanged
```

### 6.5 Stochastic Universal Sampling (SUS)

Uses a single spin of a roulette wheel with *n* equally-spaced pointers, ensuring more uniform coverage.  
**Use with:** All representations.

```cpp
#include "selection-operator/stochastic_universal_sampling.h"
StochasticUniversalSampling sel;
```

---

## 7. Benchmark Functions

These standard test functions let you evaluate how well operators and configurations perform.

| Function | Characteristics | Challenge |
|----------|----------------|-----------|
| **Sphere** | Simple unimodal bowl | Baseline — easy for most GAs |
| **Rastrigin** | Highly multimodal (many local optima on a cosine landscape) | Avoiding local optima traps |
| **Ackley** | One global minimum surrounded by many local minima | Deceptive near-optimal landscape |
| **Schwefel** | Global optimum far from local optima | Deception at large scale |
| **Rosenbrock** | Narrow, curved valley to global optimum | Slow convergence along valley |

All functions are available via the benchmark suite — see [Section 23](#23-benchmark-suite).

---

## 8. C++ API

### 8.1 Basic Usage

```cpp
#include <ga/genetic_algorithm.hpp>

double myFitness(const std::vector<double>& x) {
    // higher = better
    double sum = 0.0;
    for (double xi : x) sum += xi * xi;
    return 1000.0 / (1.0 + sum);
}

int main() {
    ga::Config cfg;
    cfg.populationSize = 60;
    cfg.generations    = 150;
    cfg.dimension      = 10;
    cfg.bounds         = {-5.0, 5.0};
    cfg.seed           = 42;

    ga::GeneticAlgorithm alg(cfg);
    ga::Result res = alg.run(myFitness);

    // Access results
    std::cout << "Best fitness: " << res.bestFitness << "\n";
    for (double g : res.bestGenes) std::cout << g << " ";
}
```

### 8.2 Customizing Operators

```cpp
#include <ga/genetic_algorithm.hpp>

ga::GeneticAlgorithm alg(cfg);

// Swap crossover operator
alg.setCrossoverOperator(ga::makeTwoPointCrossover());

// Swap mutation operator
alg.setMutationOperator(ga::makeGaussianMutation());

ga::Result res = alg.run(myFitness);
```

### 8.3 Reading Convergence History

```cpp
ga::Result res = alg.run(myFitness);

// Best fitness per generation
for (size_t i = 0; i < res.bestHistory.size(); ++i) {
    std::cout << "Gen " << i << ": best=" << res.bestHistory[i]
              << " avg=" << res.avgHistory[i] << "\n";
}
```

---

## 9. High-Level Optimizer API

`ga::api::Optimizer` provides a fluent one-call interface for single- and multi-objective optimization.

```cpp
#include <ga/api/optimizer.hpp>

ga::Config cfg;
cfg.populationSize = 40;
cfg.generations    = 80;
cfg.dimension      = 6;
cfg.bounds         = {-3.0, 3.0};

ga::api::Optimizer opt;
opt.withConfig(cfg)
   .withSeed(42)
   .withThreads(4);    // optional parallel evaluation

// Single-objective
ga::Result res = opt.optimize([](const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) s += v * v;
    return 1000.0 / (1.0 + s);
});
std::cout << "Best: " << res.bestFitness << "\n";
```

---

## 10. Python API

Python bindings are built with pybind11 and expose the full C++ interface.

### 10.1 Setup

```bash
pip install pybind11
cmake --build build -j
export PYTHONPATH=$PWD/build/python:$PYTHONPATH
```

### 10.2 Single-Objective Optimization

```python
import ga

def sphere(x):
    return 1000.0 / (1.0 + sum(v**2 for v in x))

cfg = ga.Config()
cfg.population_size = 50
cfg.generations     = 100
cfg.dimension       = 10
cfg.bounds          = ga.Bounds(-5.12, 5.12)
cfg.seed            = 42

alg = ga.GeneticAlgorithm(cfg)
result = alg.run(sphere)

print(f"Best fitness : {result.best_fitness}")
print(f"Best solution: {result.best_genes}")
```

### 10.3 Accessing Convergence History in Python

```python
for gen, (best, avg) in enumerate(zip(result.best_history, result.avg_history)):
    print(f"Gen {gen}: best={best:.4f}  avg={avg:.4f}")
```

### 10.4 NSGA-III Utilities (Python)

```python
import ga

# Generate Das-Dennis reference points for 3 objectives, 4 divisions
ref_points = ga.nsga3_reference_points(3, 4)
print(f"Reference points: {len(ref_points)}")
```

### 10.5 Checkpoint API (Python)

```python
import ga

state = ga.CheckpointState()
# Populate state fields before saving
state.config = ga.Config()   # fill with your run config
# state.result is populated from a completed run

ga.checkpoint_save_json("checkpoint.json", state)   # path first, then state
loaded = ga.checkpoint_load_json("checkpoint.json")
```

---

## 11. C API

A C-compatible interface is available for use from C code or for language interoperability.

### 11.1 Basic C Usage

```c
#include <ga/c_api.h>
#include <stdio.h>

static double sphere_fitness(const double* genes, int length, void* user_data) {
    double sum = 0.0;
    for (int i = 0; i < length; ++i)
        sum += genes[i] * genes[i];
    return 1000.0 / (1.0 + sum);
}

int main(void) {
    ga_config_c cfg = {
        .population_size = 60,
        .generations     = 100,
        .dimension       = 10,
        .crossover_rate  = 0.8,
        .mutation_rate   = 0.1,
        .lower_bound     = -5.12,
        .upper_bound     =  5.12,
        .elite_ratio     = 0.05,
        .seed            = 42
    };

    // Validate config before use
    if (ga_validate_config(&cfg) != GA_STATUS_OK) {
        fprintf(stderr, "Invalid config\n");
        return 1;
    }

    ga_handle* h = ga_create(&cfg);
    if (!h) return 1;

    ga_run(h, sphere_fitness, NULL);

    printf("Best fitness: %f\n", ga_best_fitness(h));

    // Export convergence history
    int n = ga_history_length(h);
    double history[1024];
    if (n > 0 && n <= 1024)
        ga_best_history(h, history, n);

    ga_destroy(h);
    return 0;
}
```

### 11.2 C API Functions

| Function | Description |
|----------|-------------|
| `ga_create(cfg)` | Create a GA handle with the given config. |
| `ga_validate_config(cfg)` | Check config values before running. Returns `GA_STATUS_OK` or error. |
| `ga_run(h, fitness_fn, user_data)` | Run the GA. Calls `fitness_fn` for each individual. |
| `ga_best_fitness(h)` | Return the best fitness found. |
| `ga_history_length(h)` | Number of recorded generations. |
| `ga_best_history(h, buf, n)` | Copy best-fitness-per-generation into `buf`. |
| `ga_avg_history(h, buf, n)` | Copy average-fitness-per-generation into `buf`. |
| `ga_destroy(h)` | Free all resources. |

---

## 12. Multi-Objective Optimization (NSGA-II / NSGA-III)

Optimize multiple competing objectives simultaneously and find the Pareto front.

### 12.1 NSGA-II via the High-Level API

```cpp
#include <ga/api/optimizer.hpp>

ga::api::Optimizer opt;
opt.withConfig(cfg).withSeed(42);

auto mo = opt.optimizeMultiObjective(
    {
        [](const std::vector<double>& x) {   // objective 1: minimize distance from origin
            double s = 0.0;
            for (double v : x) s += v * v;
            return s;
        },
        [](const std::vector<double>& x) {   // objective 2: minimize distance from (1,1,...)
            double s = 0.0;
            for (double v : x) s += (v - 1.0) * (v - 1.0);
            return s;
        }
    },
    /*populationSize=*/80,
    /*generations=*/80
);

std::cout << "Pareto front size: " << mo.paretoGenes.size() << "\n";
```

### 12.2 NSGA-III

Designed for problems with **three or more objectives**. Uses structured reference points (Das-Dennis method) for diversity preservation.

```cpp
auto mo3 = opt.optimizeMultiObjectiveNsga3(objectives, 80, 80, /*divisions=*/8);
std::cout << "NSGA-III Pareto points: " << mo3.paretoGenes.size() << "\n";
```

### 12.3 Direct NSGA-II API

```cpp
#include <ga/algorithms/moea/nsga2.hpp>

ga::moea::Nsga2Config ncfg;
ncfg.populationSize = 100;
ncfg.generations    = 100;
ncfg.seed           = 0;

ga::moea::Nsga2 nsga2(ncfg);
// Use nsga2.run(...) for full control
```

### 12.4 NSGA-III Reference Points

```cpp
#include <ga/moea/nsga3.hpp>

// Generate structured reference points for 3 objectives with 4 divisions
auto refPoints = ga::moea::Nsga3::generateDasDennisReferencePoints(3, 4);
```

---

## 13. Evolution Strategies (ES / CMA-ES)

Evolution Strategies are gradient-free optimization algorithms that use self-adaptive Gaussian mutations.

### 13.1 (μ, λ) and (μ + λ) Strategies

```cpp
#include <ga/es/evolution_strategies.hpp>

ga::es::EvolutionStrategyConfig cfg;
cfg.mu           = 15;     // parents
cfg.lambda       = 100;    // offspring per generation
cfg.generations  = 100;
cfg.dimension    = 10;
cfg.sigma        = 0.3;
cfg.plusStrategy = false;  // false = (mu,lambda); true = (mu+lambda)

ga::es::EvolutionStrategy es(cfg);
auto result = es.run(myFitness);
```

- **`(μ, λ)`**: Selects `μ` survivors from `λ` offspring only. The parents do not survive. Encourages exploration.
- **`(μ + λ)`**: Selects `μ` survivors from the union of `μ` parents and `λ` offspring. Elitist — the current best always survives.

### 13.2 Diagonal CMA-ES

Covariance Matrix Adaptation ES — one of the most powerful gradient-free methods for continuous optimization.

```cpp
#include <ga/es/cmaes.hpp>

ga::es::CmaEsConfig cfg;
cfg.dimension      = 10;
cfg.populationSize = 20;
cfg.sigma          = 0.5;   // initial step size

ga::es::DiagonalCmaEs cmaes(cfg);
auto result = cmaes.run(myFitness);
```

---

## 14. Genetic Programming (GP)

Evolve programs or mathematical expressions represented as trees.

> **API status:** The GP subsystem is experimental. The core building blocks are in `include/ga/gp/` — consult these headers directly for the most up-to-date interfaces, as they may change between versions.

### 14.1 Tree Nodes and Builders

Programs are represented as trees of `ga::gp::Node` primitives (functions/operators) and terminals (inputs/constants). The `TreeBuilder` class creates random trees from a configured primitive set.

Relevant headers:
- `ga/gp/node.hpp` — `Node` type and primitive definitions
- `ga/gp/tree_builder.hpp` — instance-based builder that creates and grows trees
- `ga/gp/type_system.hpp` — optional typed GP support for enforcing input/output type signatures

### 14.2 Automatically Defined Functions (ADF)

ADFs are reusable evolved sub-programs maintained alongside each individual. The ADF infrastructure is defined in:
- `ga/gp/adf.hpp` — ADF pool and registration helpers

### 14.3 Notes on API Stability

Because the GP API is still evolving, always prefer the symbols shown in the current headers in `include/ga/gp/` over any external examples. Check the header documentation and any accompanying tests under `tests/` for concrete usage patterns.

---

## 15. Parallel & Distributed Evaluation

Speed up fitness evaluation by running individuals in parallel.

### 15.1 Parallel Evaluator (Thread Pool)

The `ga::evaluation::ParallelEvaluator` is a template class that evaluates a batch of inputs concurrently.

```cpp
#include <ga/evaluation/parallel_evaluator.hpp>
#include <vector>

auto fitness = [](const std::vector<double>& x) -> double {
    double s = 0.0;
    for (double v : x) s += v * v;
    return 1000.0 / (1.0 + s);
};

// Template params: input type, output type, callable type
ga::evaluation::ParallelEvaluator<
    std::vector<double>,   // input (chromosome)
    double,                // output (fitness)
    decltype(fitness)      // callable
> evaluator(fitness, /*threads=*/4);

// The high-level Optimizer API hides this detail via withThreads()
```

### 15.2 Local Distributed Executor

Runs evaluations across threads as if they were distributed workers:

```cpp
#include <ga/evaluation/distributed_executor.hpp>

ga::LocalDistributedExecutor executor(/*workers=*/8);
```

### 15.3 Process Distributed Executor (POSIX)

Runs evaluations in separate OS processes — true parallelism even with expensive Python-bound fitness functions:

```cpp
#include <ga/evaluation/distributed_executor.hpp>

ga::ProcessDistributedExecutor executor(/*workers=*/4);
```

Build and test the process backend:

```bash
cmake --build build --target process-distributed-sanity
./build/tests/process-distributed-sanity
```

---

## 16. Checkpointing

Save and restore a run so you can continue after a crash or extend a completed run.

```cpp
#include <ga/checkpoint/checkpoint.hpp>

// Build a state from a completed run
ga::checkpoint::CheckpointState state;
state.config     = cfg;         // ga::Config used for the run
state.result     = result;      // ga::Result returned by alg.run(...)
state.generation = 100;         // current generation index
// state.rngState can store the RNG stream (as a string) for full reproducibility

// Save — CheckpointManager uses static methods; no object needed
ga::checkpoint::CheckpointManager::saveBinary("run_checkpoint.bin", state);   // fast binary
ga::checkpoint::CheckpointManager::saveJson("checkpoint.json", state);        // human-readable JSON

// Restore
ga::checkpoint::CheckpointState loaded =
    ga::checkpoint::CheckpointManager::loadBinary("run_checkpoint.bin");
// or
ga::checkpoint::CheckpointState loadedJson =
    ga::checkpoint::CheckpointManager::loadJson("checkpoint.json");
```

The saved state includes: `ga::Config`, `ga::Result` (best genes, fitness, history), generation index, and optional RNG state.

---

## 17. Adaptive Operators

Automatically tune crossover and mutation rates based on population diversity and progress.

```cpp
#include <ga/adaptive/adaptive_policy.hpp>

ga::adaptive::AdaptiveRateController controller;
ga::adaptive::AdaptiveRates rates{/*mutationRate=*/0.1, /*crossoverRate=*/0.8};

// Call once per generation, passing current diversity and improvement since last generation
rates = controller.update(rates, /*diversity=*/currentDiversity, /*bestImprovement=*/recentImprovement);

cfg.mutationRate  = rates.mutationRate;
cfg.crossoverRate = rates.crossoverRate;
```

**How it works:**
- If diversity drops and there is no improvement → mutation rate increases.
- If progress is steady and diversity is healthy → mutation rate decreases.
- This reduces the need for manual hyperparameter tuning.

---

## 18. Hybrid Optimization

Combine the global search of a GA with local refinement for faster convergence.

```cpp
#include <ga/hybrid/hybrid_optimizer.hpp>

ga::hybrid::HybridOptimizer hybrid(gaConfig);

// Define a local search function that refines a single solution in-place
auto localSearch = [](std::vector<double>& genes) {
    // Example: a few steps of gradient-free hill climbing
    for (int step = 0; step < 20; ++step) {
        std::vector<double> candidate = genes;
        // ... perturb candidate and accept if better ...
    }
};

// run(fitness, localSearch, localSearchRestarts)
// After the GA finishes, localSearch is applied to the best solution
// localSearchRestarts times (accepting improvements)
auto result = hybrid.run(myFitness, localSearch, /*localSearchRestarts=*/5);
```

**Use case:** When your GA stalls near a good solution, local search can polish it without extra generations.

---

## 19. Constraint Handling

Many real problems have feasibility requirements. The framework provides a unified `ConstraintSet` with three types of constraints.

```cpp
#include <ga/constraints/constraints.hpp>

ga::constraints::ConstraintSet cs;

// Hard constraints — return true if satisfied, false otherwise
cs.hard.push_back([](const std::vector<double>& x) {
    return x[0] + x[1] <= 5.0;   // must be satisfied
});

// Soft constraints — return a non-negative penalty amount (0 = no violation)
cs.soft.push_back([](const std::vector<double>& x) {
    return std::max(0.0, x[0] + x[1] - 5.0);   // excess amount
});

// Repair functions — fix infeasible individuals in-place
cs.repairs.push_back([](std::vector<double>& x) {
    x[0] = std::min(x[0], 5.0 - x[1]);
});
```

Use the helper functions in your fitness wrapper:

```cpp
double constrainedFitness(const std::vector<double>& x) {
    // Apply repairs first
    std::vector<double> repaired = x;
    ga::constraints::applyRepairs(repaired, cs);

    double base = myFitness(repaired);

    // Returns base fitness reduced by soft penalties; subtracts a large
    // infeasiblePenalty if any hard constraint is violated
    return ga::constraints::penalizedFitness(base, repaired, cs, /*infeasiblePenalty=*/1e6);
}
```

---

## 20. Experiment Tracking

Log all run metadata for reproducibility and comparison.

```cpp
#include <ga/tracking/experiment_tracker.hpp>

ga::tracking::ExperimentTracker tracker("my_experiment");

// Write config to a text file (key=value format)
tracker.writeConfig(cfg, "experiment_config.txt");

// After the run, write per-generation history and best solution
ga::Result result = alg.run(myFitness);
tracker.writeHistoryCSV(result, "experiment_history.csv");
tracker.writeBestSolutionCSV(result, "experiment_best.csv");
```

Output files:
- `experiment_config.txt` — run parameters (population size, rates, seed, etc.)
- `experiment_history.csv` — columns: `generation,best,average`
- `experiment_best.csv` — columns: `index,value` (one row per gene)

---

## 21. Visualization & Export

Export fitness and diversity data for plotting in Python, R, or any spreadsheet tool.

```cpp
#include <ga/visualization/export.hpp>

ga::Result result = alg.run(myFitness);

// Export fitness curve: columns generation,best,avg
ga::visualization::exportFitnessCurveCSV(result.bestHistory, result.avgHistory, "fitness_curve.csv");

// Export Pareto front (for multi-objective): one row per solution
ga::visualization::exportParetoFrontCSV(paretoObjectives, "pareto_front.csv");

// Export diversity trend: columns generation,diversity
ga::visualization::exportDiversityCSV(diversityHistory, "diversity.csv");
```

Generated CSV files have a standard schema you can load in Python:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("fitness_curve.csv")
plt.plot(df["generation"], df["best"], label="Best")
plt.plot(df["generation"], df["avg"], label="Average")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.savefig("fitness_curve.png")
```

---

## 22. Plugin Architecture

Register custom operators at runtime using the plugin registry.

`ga::plugin::Registry<Base>` is a template class — instantiate one per operator type:

```cpp
#include <ga/plugin/registry.hpp>

// The registry is an ordinary object (no singleton); own it wherever it lives
ga::plugin::Registry<CrossoverOperator> crossoverRegistry;

// Register a custom crossover by name
crossoverRegistry.registerFactory(
    "my_crossover",
    []() { return std::make_unique<MyCrossover>(); }
);

// Check if registered
if (crossoverRegistry.has("my_crossover")) {
    // Create an instance by name
    auto op = crossoverRegistry.create("my_crossover");
    alg.setCrossoverOperator(std::move(op));
}

// List all registered names
for (const auto& name : crossoverRegistry.names()) {
    std::cout << name << "\n";
}
```

---

## 23. Benchmark Suite

Test operator speed, convergence quality, and scalability.

### 23.1 Build the Benchmark Executable

```bash
cmake --build build -j
```

### 23.2 Run Benchmarks

```bash
# Run everything
./build/bin/ga-benchmark --all

# Individual categories
./build/bin/ga-benchmark --operators     # crossover/mutation/selection speed
./build/bin/ga-benchmark --functions     # convergence on Sphere, Rastrigin, Ackley, etc.
./build/bin/ga-benchmark --scalability   # performance vs. population size and dimension

# Custom iteration count
./build/bin/ga-benchmark --operators --iterations 2000

# Export results to CSV
./build/bin/ga-benchmark --all --csv

# Show help
./build/bin/ga-benchmark --help
```

### 23.3 Benchmark Output

The tool creates:
- **Console output** – real-time progress and a summary table.
- **`benchmark_results.txt`** – full detailed report.
- **`benchmark_results.csv`** – machine-readable (with `--csv`).

### 23.4 Typical Operator Throughput

| Operator | Throughput |
|----------|-----------|
| Gaussian Mutation | ~6.6 M ops/sec |
| Swap Mutation | ~20 M ops/sec |
| Blend Crossover (BLX-α) | ~5 M ops/sec |
| Two-Point Crossover | ~2 M ops/sec |
| Tournament Selection | ~181 K ops/sec |

---

## 24. Building the Project

### Prerequisites

- CMake ≥ 3.16
- C++17 compiler (GCC 7+, Clang 5+, or MSVC 2017+)

### Build Steps

```bash
# Clone the repository
git clone <repo-url>
cd Genetic_algorithm

# Configure and build
mkdir build && cd build
cmake ..
cmake --build .   # add -j<N> for parallel jobs, e.g. -j4
```

### Using the Build Script

```bash
./build.sh            # build
./build.sh --run      # build and run the demo
./build.sh --debug    # debug build
./build.sh --clean    # clean and rebuild
./build.sh --install  # install to system
```

### Build Targets

```bash
cmake --build build --target simple-ga-test           # interactive demo
cmake --build build --target c-api-sanity             # C API test
cmake --build build --target features-foundation-sanity  # foundation sanity test
cmake --build build --target nsga3-sanity             # NSGA-III sanity test
cmake --build build --target process-distributed-sanity  # distributed backend test
```

---

## 25. Running the Interactive Demo

```bash
./build/bin/simple_ga_test
```

The demo guides you through:

1. **Representation** – choose `binary`, `real_valued`, `integer`, or `permutation`.
2. **Crossover operator** – validated against your chosen representation.
3. **Mutation operator** – validated against your chosen representation.
4. **Selection operator** – choose `tournament`, `roulette`, `rank`, etc.

### Scripted Examples

```bash
# Real-valued with Blend crossover and Gaussian mutation
echo -e "real_valued\nblend\ngaussian\ntournament" | ./build/bin/simple_ga_test

# Binary with Uniform crossover and Bit-flip mutation
echo -e "binary\nuniform\nbit_flip\ntournament" | ./build/bin/simple_ga_test

# Integer with Arithmetic crossover and Creep mutation
echo -e "integer\narithmetic\ncreep\ntournament" | ./build/bin/simple_ga_test

# Permutation (TSP-style) with Order crossover and Swap mutation
echo -e "permutation\norder_crossover\nswap\ntournament" | ./build/bin/simple_ga_test
```

---

## 26. Troubleshooting

| Problem | Solution |
|---------|---------|
| `cmake` not found | `sudo apt install cmake` (Linux) or `brew install cmake` (macOS) |
| Compiler not found | `sudo apt install build-essential` (Linux) or `xcode-select --install` (macOS) |
| Build errors | Delete the `build/` folder and start fresh: `rm -rf build && mkdir build && cd build && cmake .. && cmake --build .` |
| Python `import ga` fails | Run `export PYTHONPATH=$PWD/build/python:$PYTHONPATH` before importing |
| Non-deterministic results | Set `cfg.seed` to a fixed non-zero integer |
| Premature convergence | Increase `populationSize`, raise `mutationRate`, or lower `eliteRatio` |
| Slow convergence | Increase `crossoverRate` or switch to a stronger crossover (e.g., BLX-α for real-valued) |
| Fitness never improves | Check that your fitness function returns **higher values for better solutions** (convert minimization with `1/(1+f)`) |

---

*For architecture details and internal design decisions, see [ARCHITECTURE.md](ARCHITECTURE.md). For a full feature roadmap, see [FEATURE_CHECKLIST.md](FEATURE_CHECKLIST.md).*
