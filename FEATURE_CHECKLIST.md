# Genetic Algorithm Library - Complete Technical Checklist

This document is a practical, implementation-oriented checklist for evolving this repository to DEAP-level capability and beyond, while staying clean, reusable, and language-export friendly.

## 1) Status Snapshot

Legend: implemented, partial, planned

| Area | Status | Notes in current repo |
|---|---|---|
| Core vector GA engine | implemented | `include/ga/genetic_algorithm.hpp`, `src/genetic_algorithm.cpp` |
| Crossover operators (many) | implemented | `crossover/*.h`, `crossover/*.cc` |
| Mutation operators (many) | implemented | `mutation/*.h`, `mutation/*.cc` |
| Selection operators (classic) | implemented | `selection-operator/*.h`, `selection-operator/*.cc` |
| Basic tree node + subtree crossover | partial | subtree crossover exists; typed GP node system/tree builder/ADF pool now added in `include/ga/gp/*` |
| ES/CMA-ES | partial | self-adaptive mutation exists; now includes `(mu,lambda)/(mu+lambda)` ES and diagonal CMA-ES in `include/ga/es/*` |
| Multi-objective (NSGA-II/III, SPEA2) | partial | NSGA-II run loop + NSGA-III reference-point niching with intercept-based normalization + SPEA2 + MO-CMA-ES wrappers added |
| Co-evolution | partial | cooperative/competitive multi-population engine foundation added in `include/ga/coevolution/coevolution.hpp` |
| Parallel/distributed evaluation | partial | serial/parallel evaluators plus local distributed executor, process-based distributed executor (POSIX), and backend interface added in `include/ga/evaluation/*` |
| Checkpointing | partial | binary + JSON checkpoint save/load manager added in `include/ga/checkpoint/checkpoint.hpp` |
| Adaptive operators | partial | adaptive rate controller added in `include/ga/adaptive/adaptive_policy.hpp` |
| Hybrid optimization | partial | GA+local-search hybrid pipeline added in `include/ga/hybrid/hybrid_optimizer.hpp` |
| Constraint handling | partial | hard/soft/repair constraint toolkit added in `include/ga/constraints/constraints.hpp` |
| Experiment tracking | partial | C API history + experiment tracker outputs added in `include/ga/tracking/experiment_tracker.hpp` |
| Visualization support | partial | CSV export helpers for fitness/pareto/diversity added in `include/ga/visualization/export.hpp` |
| Plugin architecture | partial | generic plugin/factory registry added in `include/ga/plugin/registry.hpp` |
| Benchmark suite | implemented | `benchmark/ga_benchmark.h`, `benchmark/ga_benchmark.cc` |
| Reproducibility controls | implemented | seed-based RNG support in config |
| High-level API | partial | fluent optimizer API now added for single objective + NSGA-II + NSGA-III workflows in `include/ga/api/optimizer.hpp` |
| C and Python export path | partial | Python bindings present (now includes NSGA-III utils + checkpoint JSON APIs); baseline C ABI wrapper added (`include/ga/c_api.h`, `src/c_api.cpp`) |

## 2) Complete Checklist (What, Why, How)

## 2.1 Core Representations

- What to include:
  - List, array, set, dictionary, tree, and ndarray-like representations
- Why it matters:
  - Different encodings map naturally to different domains (TSP, scheduling, symbolic regression, feature subsets, graph-like structures)
- How it works:
  - Introduce a stable genome abstraction and concrete genome types.
  - Suggested core contract:

```cpp
class IGenome {
public:
    virtual ~IGenome() = default;
    virtual std::unique_ptr<IGenome> clone() const = 0;
    virtual std::string encodingName() const = 0;
};
```

  - Concrete types: `VectorGenome<double>`, `VectorGenome<int>`, `BitsetGenome`, `PermutationGenome`, `TreeGenome`, `SetGenome`, `MapGenome`, `TensorGenome`.

## 2.2 Genetic Programming (Tree-Based)

- What to include:
  - Prefix/expression trees with function and terminal sets
- Why it matters:
  - Required for symbolic regression and program synthesis
- How it works:
  - Represent `(+ (* x y) 3)` as a typed tree.
  - Use subtree crossover and subtree mutation.
  - Add depth/size limits to prevent tree bloat.

## 2.3 Strongly Typed and Loosely Typed GP

- What to include:
  - Strongly typed GP and loosely typed GP
- Why it matters:
  - Strong typing prevents invalid programs; loose typing broadens exploration
- How it works:
  - Strong typing: enforce return/input type signatures at generation and variation time.
  - Loose typing: allow generic node compatibility with runtime checks or coercion policies.

## 2.4 Automatically Defined Functions (ADF)

- What to include:
  - Evolved reusable sub-functions
- Why it matters:
  - Supports modularity, reusability, and scalable search
- How it works:
  - Maintain function slots (ADF0..ADFk) in each individual or population-wide library.
  - Individuals may call ADF nodes during evaluation.
  - Co-evolve caller trees and ADF bodies.

## 2.5 Evolution Strategies (ES)

- What to include:
  - `(mu, lambda)` and `(mu + lambda)`
  - CMA-ES
- Why it matters:
  - Strong performance for continuous optimization
- How it works:
  - ES:
    - Sample offspring via Gaussian perturbation using strategy parameters.
    - Survivor policy differs for comma vs plus strategy.
  - CMA-ES:
    - Keep mean vector, covariance matrix, and step size.
    - Sample from multivariate Gaussian and update from weighted elites.

## 2.6 Multi-Objective Optimization

- What to include:
  - NSGA-II, NSGA-III, SPEA2, MO-CMA-ES
- Why it matters:
  - Most practical problems optimize multiple competing objectives
- How it works:
  - Non-dominated sorting and Pareto front maintenance.
  - Diversity preservation (crowding distance / reference points).
  - Archive support for external elite set.

## 2.7 Co-Evolution (Multi-Population)

- What to include:
  - Cooperative and competitive co-evolution
- Why it matters:
  - Useful for modular decomposition and adversarial domains
- How it works:
  - Maintain multiple populations with separate or shared operators.
  - Cross-population fitness evaluations (predator-prey, generator-discriminator patterns).

## 2.8 Parallel and Distributed Evaluation

- What to include:
  - Threaded and process-based evaluation, optional distributed execution
- Why it matters:
  - Fitness evaluation dominates runtime cost
- How it works:
  - Abstract evaluator backend:
    - single-thread executor
    - thread-pool executor
    - process executor
    - remote executor (future)
  - Keep deterministic mode by controlling task order and RNG streams.

## 2.9 Checkpointing

- What to include:
  - Snapshot and restore of run state
- Why it matters:
  - Crash recovery and long-running experiment continuation
- How it works:
  - Persist:
    - population genomes and fitness/objectives
    - generation index
    - RNG state
    - operator/adaptation state
  - Store in binary for speed and JSON for inspectability.

## 2.10 Adaptive Genetic Operators (Beyond DEAP)

- What to include:
  - Dynamic mutation and crossover rates
- Why it matters:
  - Reduces manual hyperparameter tuning
- How it works:
  - Introduce adaptation policy based on diversity and progress metrics.
  - Example rule:
    - if diversity drops and no improvement, increase mutation
    - if stable improvement and healthy diversity, decrease mutation

## 2.11 Hybrid Optimization (Beyond DEAP)

- What to include:
  - GA + local search, GA + PSO/DE
- Why it matters:
  - Speeds convergence and improves final solution quality
- How it works:
  - Memetic loop:
    - run variation and selection
    - apply local search to top-k individuals
  - Hybrid scheduler controls when and how many local refinements run.

## 2.12 Constraint Handling

- What to include:
  - Hard constraints, soft penalties, repair operators
- Why it matters:
  - Real optimization problems include feasibility requirements
- How it works:
  - Unified feasibility API:
    - feasibility check
    - penalty strategy
    - repair strategy
  - Support both static and adaptive penalties.

## 2.13 Experiment Tracking (Beyond DEAP)

- What to include:
  - Config, metrics, artifacts, and run metadata logs
- Why it matters:
  - Reproducibility, debugging, and fair comparisons
- How it works:
  - Persist run manifest:
    - seed, config, operator choices
    - per-generation metrics
    - best solutions and checkpoints

## 2.14 Visualization (Beyond DEAP)

- What to include:
  - Fitness curve, Pareto front, diversity trend plots
- Why it matters:
  - Makes algorithm dynamics visible and debuggable
- How it works:
  - Export canonical CSV/JSON schema.
  - Provide Python plotting helper package consuming exported metrics.

## 2.15 Plugin Architecture

- What to include:
  - Stable interfaces and registry for user-defined operators
- Why it matters:
  - Third-party extensibility without modifying core
- How it works:
  - Interface-based contracts for selection/crossover/mutation/replacement/adaptation.
  - Runtime registry by string key + factory.
  - Versioned plugin metadata.

## 2.16 Benchmark Suite

- What to include:
  - Sphere, Rastrigin, Ackley, Rosenbrock, Schwefel and constrained/MOO test sets
- Why it matters:
  - Objective performance and regression measurement
- How it works:
  - Standardized benchmark harness with fixed seeds and report schema.

## 2.17 Reproducibility

- What to include:
  - Global and component-level seed control; deterministic modes
- Why it matters:
  - Scientific reliability and debugging
- How it works:
  - Split RNG streams by subsystem (init, variation, selection, evaluation).
  - Serialize RNG states in checkpoints.

## 2.18 High-Level API (Differentiator)

- What to include:
  - One-call optimization API with sensible defaults
- Why it matters:
  - Faster onboarding than lower-level frameworks
- How it works:
  - Pipeline builder API:

```cpp
auto result = ga::Optimizer()
    .problem(problem)
    .representation("real_vector")
    .algorithm("nsga2")
    .seed(42)
    .optimize();
```

## 3) Target Folder Structure (C++ Core First)

```text
include/ga/
  core/
    individual.hpp
    population.hpp
    problem.hpp
    evaluator.hpp
    engine.hpp
    result.hpp
    config.hpp
  representations/
    vector_genome.hpp
    permutation_genome.hpp
    bitset_genome.hpp
    tree_genome.hpp
    set_genome.hpp
    map_genome.hpp
    tensor_genome.hpp
  operators/
    selection/
    crossover/
    mutation/
    replacement/
    adaptation/
  algorithms/
    ga/
    es/
    cmaes/
    moea/
      nsga2/
      nsga3/
      spea2/
  gp/
    node.hpp
    type_system.hpp
    tree_builder.hpp
    adf.hpp
  constraints/
    feasibility.hpp
    penalty.hpp
    repair.hpp
  checkpoint/
    serializer.hpp
    checkpoint_store.hpp
  tracking/
    tracker.hpp
    metric_schema.hpp
  benchmark/
  api/
    optimizer.hpp
    builder.hpp

src/ga/...
src/es/...
src/moea/...
src/gp/...

bindings/
  c/
    ga_c_api.h
    ga_c_api.cpp
  python/
    ga_bindings.cpp
```

## 4) Core Class Architecture (Ready to Implement)

```cpp
namespace ga {

struct Evaluation {
    std::vector<double> objectives;
    bool feasible = true;
    double penalty = 0.0;
};

class IGenome {
public:
    virtual ~IGenome() = default;
    virtual std::unique_ptr<IGenome> clone() const = 0;
    virtual std::string encodingName() const = 0;
};

class Individual {
public:
    std::unique_ptr<IGenome> genome;
    Evaluation eval;
    int age = 0;
};

class IProblem {
public:
    virtual ~IProblem() = default;
    virtual size_t objectiveCount() const = 0;
    virtual Evaluation evaluate(const IGenome& genome) const = 0;
};

class ISelection {
public:
    virtual ~ISelection() = default;
    virtual std::vector<size_t> select(const std::vector<Individual>& pop, size_t count) = 0;
};

class ICrossover {
public:
    virtual ~ICrossover() = default;
    virtual std::pair<std::unique_ptr<IGenome>, std::unique_ptr<IGenome>>
    apply(const IGenome& p1, const IGenome& p2) = 0;
};

class IMutation {
public:
    virtual ~IMutation() = default;
    virtual void apply(IGenome& genome) = 0;
};

class IAlgorithm {
public:
    virtual ~IAlgorithm() = default;
    virtual void step() = 0;
};

} // namespace ga
```

## 5) C, C++, Python Export Strategy

- C++:
  - Primary source of truth and fastest path for new features.
- C API:
  - Expose stable opaque handles and plain structs for ABI safety.
  - Keep C ABI minimal and versioned.
- Python:
  - Bind C++ API with pybind11.
  - Expose high-level `optimize()` and optional low-level operator APIs.

## 6) Delivery Plan (Practical Sequence)

1. Foundation refactor:
   - introduce `IGenome`, `IProblem`, and algorithm-agnostic `Individual`
2. MOEA core:
   - NSGA-II + Pareto archive + metrics export
3. ES family:
   - `(mu, lambda)`, `(mu + lambda)`, then CMA-ES
4. GP system:
   - typed trees, subtree mutation, then ADF
5. Runtime systems:
   - parallel evaluators, checkpointing, tracking
6. Beyond-DEAP differentiators:
   - adaptive operators, hybrid local search, ergonomic optimizer API
7. Language exports:
   - C wrapper and expanded Python bindings

## 7) Definition of Done for "DEAP-Level + Beyond"

- DEAP parity:
  - multiple encodings, GP, ES, MOEA, co-evolution, parallel execution, checkpoints
- Beyond DEAP:
  - adaptive operators, hybrid optimization, built-in experiment tracking, first-class visualization export, one-call high-level API
- Engineering quality:
  - deterministic mode, benchmark regressions, unit/integration tests, documented extension points
