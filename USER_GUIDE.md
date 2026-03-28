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
./build/bin/minimal
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

The framework supports four chromosome types. Choose the one that matches your problem.

### 3.1 Binary (`BitString`)

A vector of `bool` values (`0` or `1`).  
**Best for:** feature selection, binary optimization, combinatorial problems where choices are yes/no.

```cpp
// Operators compatible with binary: One-point, Two-point, Uniform crossover; Bit-flip mutation
```

### 3.2 Real-Valued (`RealVector`)

A vector of `double` values within configurable bounds.  
**Best for:** continuous function optimization, parameter tuning, neural network weights.

```cpp
// Compatible operators: Arithmetic, Blend (BLX-α), SBX, One-point, Two-point, Uniform crossover;
//                        Gaussian, Uniform mutation
```

### 3.3 Integer (`IntVector`)

A vector of integer values.  
**Best for:** discrete scheduling, resource allocation, index-based problems.

```cpp
// Compatible operators: One-point, Two-point, Uniform, Arithmetic crossover;
//                        Random resetting, Creep mutation
```

### 3.4 Permutation (`Permutation`)

A vector of integers where each value appears exactly once — an ordering.  
**Best for:** Traveling Salesman Problem (TSP), job scheduling, route optimization.

```cpp
// Compatible operators: Order (OX), Partially Mapped (PMX), Cycle (CX) crossover;
//                        Swap, Insert, Scramble, Inversion mutation
```

---

## 4. Crossover Operators

Crossover combines two parent chromosomes to produce offspring. Select the operator that suits your representation.

### 4.1 One-Point Crossover

Splits both parents at a single random point and swaps their tails.  
**Use with:** Binary, Real-valued, Integer.

```cpp
alg.setCrossoverOperator(ga::makeOnePointCrossover());
```

### 4.2 Two-Point Crossover

Two random cut points divide the chromosome into three segments; the middle segment is swapped.  
**Use with:** Binary, Real-valued, Integer.

```cpp
alg.setCrossoverOperator(ga::makeTwoPointCrossover());
```

### 4.3 Uniform Crossover

Each gene is independently inherited from either parent with equal probability.  
**Use with:** Binary, Real-valued, Integer.  
**Tip:** Produces more diversity than point-based crossovers.

```cpp
alg.setCrossoverOperator(ga::makeUniformCrossover());
```

### 4.4 Arithmetic Crossover

Each child gene is a weighted average of the two parent genes: `α·p1 + (1−α)·p2`.  
**Use with:** Real-valued, Integer.

```cpp
alg.setCrossoverOperator(ga::makeArithmeticCrossover());
```

### 4.5 Blend Crossover (BLX-α)

Extends the search range slightly beyond the parents, exploring values around them.  
**Use with:** Real-valued.  
**Tip:** α = 0.5 is a common default, giving good exploration.

```cpp
alg.setCrossoverOperator(ga::makeBlendCrossover(/*alpha=*/0.5));
```

### 4.6 Simulated Binary Crossover (SBX)

Mimics the behavior of one-point crossover for binary strings in real-valued space.  
**Use with:** Real-valued.  
**Tip:** Widely used in NSGA-II and other evolutionary algorithms.

```cpp
alg.setCrossoverOperator(ga::makeSimulatedBinaryCrossover(/*eta=*/2.0));
```

### 4.7 Order Crossover (OX)

Preserves the relative order of elements from one parent while filling the rest from the other.  
**Use with:** Permutation.

```cpp
alg.setCrossoverOperator(ga::makeOrderCrossover());
```

### 4.8 Partially Mapped Crossover (PMX)

Preserves absolute position information by creating a partial mapping between the parents.  
**Use with:** Permutation.

```cpp
alg.setCrossoverOperator(ga::makePartiallyMappedCrossover());
```

### 4.9 Cycle Crossover (CX)

Identifies cycles of corresponding positions between parents and alternates which parent supplies each cycle.  
**Use with:** Permutation.

```cpp
alg.setCrossoverOperator(ga::makeCycleCrossover());
```

### 4.10 Differential Evolution Crossover

Perturbs a base individual using the difference of two others. Often used with the DE/rand/1 strategy.  
**Use with:** Real-valued.

```cpp
alg.setCrossoverOperator(ga::makeDifferentialEvolutionCrossover(/*F=*/0.8, /*CR=*/0.9));
```

### 4.11 Multi-Point Crossover

Generalization of one- and two-point crossover: uses *k* random cut points.  
**Use with:** Binary, Real-valued, Integer.

```cpp
alg.setCrossoverOperator(ga::makeMultiPointCrossover(/*k=*/3));
```

### 4.12 Uniform K-Vector Crossover

Applies uniform crossover independently to each gene dimension using a separate probability mask.  
**Use with:** Real-valued.

```cpp
alg.setCrossoverOperator(ga::makeUniformKVectorCrossover());
```

### 4.13 Edge Assembly Crossover (EAX)

Preserves edges (adjacency pairs) from both parents.  
**Use with:** Permutation (graph/TSP problems).

```cpp
alg.setCrossoverOperator(ga::makeEdgeCrossover());
```

### 4.14 Cut-and-Crossfill Crossover

Cuts at a random point; the remaining genes are filled from the other parent in their original order.  
**Use with:** Permutation.

```cpp
alg.setCrossoverOperator(ga::makeCutAndCrossfillCrossover());
```

### 4.15 Line Recombination

Offspring are placed on the line connecting the two parents in gene space.  
**Use with:** Real-valued.

```cpp
alg.setCrossoverOperator(ga::makeLineRecombination());
```

### 4.16 Intermediate Recombination

Each offspring gene is a random blend between the corresponding parent genes.  
**Use with:** Real-valued.

```cpp
alg.setCrossoverOperator(ga::makeIntermediateRecombination());
```

### 4.17 Diploid Recombination

Maintains a dominant and recessive copy of each gene; the expressed value follows dominance rules.  
**Use with:** Binary.

```cpp
alg.setCrossoverOperator(ga::makeDiploidRecombination());
```

### 4.18 Subtree Crossover

Swaps randomly selected subtrees between two tree-based individuals.  
**Use with:** Genetic Programming (tree representation).

```cpp
alg.setCrossoverOperator(ga::makeSubtreeCrossover());
```

---

## 5. Mutation Operators

Mutation randomly modifies one or more genes to maintain diversity and avoid premature convergence.

### 5.1 Bit-Flip Mutation

Flips each bit independently with probability `mutationRate`.  
**Use with:** Binary.

```cpp
alg.setMutationOperator(ga::makeBitFlipMutation());
```

### 5.2 Gaussian Mutation

Adds a small Gaussian random noise to each gene. Keeps the search local.  
**Use with:** Real-valued.

```cpp
alg.setMutationOperator(ga::makeGaussianMutation());
```

### 5.3 Uniform Mutation

Replaces a gene with a uniformly random value within bounds.  
**Use with:** Real-valued.

```cpp
alg.setMutationOperator(ga::makeUniformMutation());
```

### 5.4 Swap Mutation

Picks two random positions and swaps their values.  
**Use with:** Permutation.

```cpp
alg.setMutationOperator(ga::makeSwapMutation());
```

### 5.5 Insert Mutation

Removes a gene from a random position and inserts it at another random position.  
**Use with:** Permutation.

```cpp
alg.setMutationOperator(ga::makeInsertMutation());
```

### 5.6 Scramble Mutation

Selects a random subset of genes and shuffles them in place.  
**Use with:** Permutation.

```cpp
alg.setMutationOperator(ga::makeScrambleMutation());
```

### 5.7 Inversion Mutation

Reverses the order of genes between two random positions.  
**Use with:** Permutation.

```cpp
alg.setMutationOperator(ga::makeInversionMutation());
```

### 5.8 Creep Mutation

Adds or subtracts a small constant (creep) to an integer gene.  
**Use with:** Integer.

```cpp
alg.setMutationOperator(ga::makeCreepMutation(/*step=*/1));
```

### 5.9 Random Resetting Mutation

Replaces a gene with a randomly chosen integer from a valid range.  
**Use with:** Integer.

```cpp
alg.setMutationOperator(ga::makeRandomResettingMutation());
```

### 5.10 Self-Adaptive Mutation

Each individual carries its own mutation step-size, which also evolves alongside the genes. This is the foundation of Evolution Strategies.  
**Use with:** Real-valued (ES contexts).

```cpp
alg.setMutationOperator(ga::makeSelfAdaptiveMutation());
```

### 5.11 List Mutation

General mutation for list-based genomes (insert, delete, or replace list elements).  
**Use with:** Variable-length representations.

```cpp
alg.setMutationOperator(ga::makeListMutation());
```

---

## 6. Selection Operators

Selection decides which individuals survive or reproduce based on their fitness.

### 6.1 Tournament Selection

Randomly samples *k* individuals and picks the best among them. A common default.  
**Use with:** All representations.  
**Tip:** Higher tournament size (*k*) → stronger selection pressure.

```cpp
alg.setSelectionOperator(ga::makeTournamentSelection(/*k=*/3));
```

### 6.2 Roulette Wheel Selection (Fitness-Proportionate)

Each individual's probability of selection is proportional to its fitness.  
**Use with:** All representations.  
**Caution:** Sensitive to fitness scaling; a dominant individual can take over quickly.

```cpp
alg.setSelectionOperator(ga::makeRouletteWheelSelection());
```

### 6.3 Rank Selection

Individuals are ranked by fitness; selection probability is proportional to rank, not raw fitness. More stable than roulette wheel.  
**Use with:** All representations.

```cpp
alg.setSelectionOperator(ga::makeRankSelection());
```

### 6.4 Elitism Selection

The top fraction of individuals (controlled by `eliteRatio`) is copied unchanged into the next generation.  
**Use with:** All representations (built into the GA engine automatically).

```cpp
// Configured via Config; no manual operator set needed
cfg.eliteRatio = 0.10;   // top 10% survive unchanged
```

### 6.5 Stochastic Universal Sampling (SUS)

Uses a single spin of a roulette wheel with *n* equally-spaced pointers, ensuring more uniform coverage.  
**Use with:** All representations.

```cpp
alg.setSelectionOperator(ga::makeStochasticUniversalSampling());
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
    return 1000.0 / (1.0 + sum(xi**2 for xi in x))

cfg = ga.Config()
cfg.populationSize = 50
cfg.generations    = 100
cfg.dimension      = 10
cfg.bounds         = ga.Bounds(-5.12, 5.12)
cfg.seed           = 42

alg = ga.GeneticAlgorithm(cfg)
result = alg.run(sphere)

print(f"Best fitness : {result.bestFitness}")
print(f"Best solution: {result.bestGenes}")
```

### 10.3 Accessing Convergence History in Python

```python
for gen, (best, avg) in enumerate(zip(result.bestHistory, result.avgHistory)):
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
ga.checkpoint_save_json(state, "checkpoint.json")
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
        .populationSize = 60,
        .generations    = 100,
        .dimension      = 10,
        .crossoverRate  = 0.8,
        .mutationRate   = 0.1,
        .lowerBound     = -5.12,
        .upperBound     =  5.12,
        .eliteRatio     = 0.05,
        .seed           = 42
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
auto refPoints = ga::moea::generateReferencePoints(3, 4);
```

---

## 13. Evolution Strategies (ES / CMA-ES)

Evolution Strategies are gradient-free optimization algorithms that use self-adaptive Gaussian mutations.

### 13.1 (μ, λ) and (μ + λ) Strategies

```cpp
#include <ga/es/evolution_strategies.hpp>

ga::es::EsConfig cfg;
cfg.mu     = 15;    // parents
cfg.lambda = 100;   // offspring per generation
cfg.strategy = ga::es::Strategy::CommaSelection;   // (mu, lambda)
// cfg.strategy = ga::es::Strategy::PlusSelection; // (mu + lambda)

ga::es::EvolutionStrategy es(cfg);
auto result = es.run(myFitness);
```

- **`(μ, λ)`**: Selects `μ` survivors from `λ` offspring only. The parents do not survive. Encourages exploration.
- **`(μ + λ)`**: Selects `μ` survivors from the union of `μ` parents and `λ` offspring. Elitist — the current best always survives.

### 13.2 Diagonal CMA-ES

Covariance Matrix Adaptation ES — one of the most powerful gradient-free methods for continuous optimization.

```cpp
#include <ga/es/cmaes.hpp>

ga::es::CmaesConfig cfg;
cfg.dimension   = 10;
cfg.populationSize = 20;
cfg.sigma0      = 0.5;   // initial step size

ga::es::Cmaes cmaes(cfg);
auto result = cmaes.run(myFitness);
```

---

## 14. Genetic Programming (GP)

Evolve programs or mathematical expressions represented as trees.

### 14.1 Tree Nodes

```cpp
#include <ga/gp/node.hpp>
#include <ga/gp/tree_builder.hpp>

// Define the function set (operators) and terminal set (variables, constants)
ga::gp::FunctionSet funcs = {"+", "-", "*", "/"};
ga::gp::TerminalSet terms = {"x", "y", "1.0"};

// Build a random initial tree of depth ≤ 4
auto tree = ga::gp::TreeBuilder::grow(funcs, terms, /*maxDepth=*/4);
```

### 14.2 Type System

```cpp
#include <ga/gp/type_system.hpp>

// Enforce type safety in function/terminal signatures
ga::gp::TypedNode node("sin", {ga::gp::Type::Real}, ga::gp::Type::Real);
```

### 14.3 Automatically Defined Functions (ADF)

Reusable evolved sub-programs that individuals can call:

```cpp
#include <ga/gp/adf.hpp>

ga::gp::AdfPool pool;
pool.define("ADF0", mySubtree);   // register a reusable function
// Individuals can now call ADF0 as if it were a primitive
```

---

## 15. Parallel & Distributed Evaluation

Speed up fitness evaluation by running individuals in parallel.

### 15.1 Parallel Evaluator (Thread Pool)

```cpp
#include <ga/evaluation/parallel_evaluator.hpp>

ga::ParallelEvaluator evaluator(/*threads=*/4);
// The optimizer API uses this automatically via withThreads()
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

// Save current run state
ga::checkpoint::CheckpointManager mgr("run_checkpoint");
mgr.save(state);             // binary format (fast)
mgr.saveJson(state, "checkpoint.json");  // JSON format (human-readable)

// Restore state
ga::checkpoint::CheckpointState loaded;
mgr.load(loaded);
// or
mgr.loadJson(loaded, "checkpoint.json");
```

The saved state includes: population genomes, fitness values, generation index, and RNG state.

---

## 17. Adaptive Operators

Automatically tune crossover and mutation rates based on population diversity and progress.

```cpp
#include <ga/adaptive/adaptive_policy.hpp>

ga::adaptive::AdaptivePolicy policy;
policy.setDiversityThreshold(0.1);   // increase mutation if diversity drops below this
policy.setImprovementWindow(20);     // look at last 20 generations for improvement

// The policy is queried each generation to update rates
double mutRate = policy.mutationRate(currentDiversity, recentImprovement);
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
hybrid.setLocalSearchSteps(50);    // run 50 steps of local search on top-k individuals
hybrid.setLocalSearchK(5);         // refine the top 5 individuals per generation

auto result = hybrid.run(myFitness);
```

**Use case:** When your GA stalls near a good solution, local search can polish it without extra generations.

---

## 19. Constraint Handling

Many real problems have feasibility requirements. The framework provides three approaches:

### 19.1 Hard Constraints

Reject infeasible solutions outright:

```cpp
#include <ga/constraints/constraints.hpp>

ga::constraints::HardConstraint c;
c.addConstraint([](const std::vector<double>& x) {
    return x[0] + x[1] <= 5.0;   // must be satisfied
});
```

### 19.2 Soft Constraints (Penalty)

Penalize infeasible solutions by subtracting from fitness:

```cpp
ga::constraints::PenaltyConstraint p;
p.setPenaltyFactor(100.0);   // large factor → stronger pressure toward feasibility
p.addViolation([](const std::vector<double>& x) {
    return std::max(0.0, x[0] + x[1] - 5.0);   // excess amount
});
```

### 19.3 Repair Operator

Automatically fix infeasible individuals before evaluation:

```cpp
ga::constraints::RepairOperator repair;
repair.setRepairFn([](std::vector<double>& x) {
    // Clip values to feasible region
    x[0] = std::min(x[0], 5.0 - x[1]);
});
```

---

## 20. Experiment Tracking

Log all run metadata for reproducibility and comparison.

```cpp
#include <ga/tracking/experiment_tracker.hpp>

ga::tracking::ExperimentTracker tracker("my_experiment");
tracker.logConfig(cfg);
tracker.logGeneration(gen, bestFitness, avgFitness);
tracker.logBestSolution(result.bestGenes, result.bestFitness);
tracker.save("experiment_results.json");
```

The saved log includes: config, seed, operator choices, per-generation metrics, and the best solution found.

---

## 21. Visualization & Export

Export fitness and diversity data for plotting in Python, R, or any spreadsheet tool.

```cpp
#include <ga/visualization/export.hpp>

// Export fitness curve
ga::visualization::exportFitnessCsv(result, "fitness_curve.csv");

// Export Pareto front (for multi-objective)
ga::visualization::exportParetoCsv(paretoObjectives, "pareto_front.csv");

// Export diversity trend
ga::visualization::exportDiversityCsv(diversityHistory, "diversity.csv");
```

Generated CSV files have a standard schema you can load in Python:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("fitness_curve.csv")
plt.plot(df["generation"], df["best_fitness"], label="Best")
plt.plot(df["generation"], df["avg_fitness"], label="Average")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.savefig("fitness_curve.png")
```

---

## 22. Plugin Architecture

Register custom operators at runtime using the plugin registry.

```cpp
#include <ga/plugin/registry.hpp>

// Register a custom crossover by name
ga::plugin::Registry::instance().registerCrossover(
    "my_crossover",
    []() { return std::make_unique<MyCrossover>(); }
);

// Look up and use by name
auto op = ga::plugin::Registry::instance().createCrossover("my_crossover");
```

This allows third-party operators without modifying the core library.

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
