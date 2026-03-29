# Python API Reference for Genetic Algorithm Framework

This document provides a comprehensive reference of all important APIs exposed to Python through the `ga` module.

**Last Updated:** 2026-03-29
**Module Version:** Based on ga_bindings.cpp
**Total Classes:** 52
**Total Functions:** 26

---

## Table of Contents

1. [Core Configuration and Result Classes](#1-core-configuration-and-result-classes)
2. [Genome Representation Classes](#2-genome-representation-classes)
3. [Genetic Programming Classes](#3-genetic-programming-classes)
4. [Main Algorithm Classes](#4-main-algorithm-classes)
5. [Multi-Objective Evolutionary Algorithms (MOEA)](#5-multi-objective-evolutionary-algorithms-moea)
6. [Evolution Strategies (ES/CMA-ES/MO-CMA-ES)](#6-evolution-strategies-escma-esmo-cma-es)
7. [Constraints and Adaptive Control](#7-constraints-and-adaptive-control)
8. [Hybrid and Coevolution](#8-hybrid-and-coevolution)
9. [Parallel Evaluation](#9-parallel-evaluation)
10. [Tracking, Visualization, and Checkpointing](#10-tracking-visualization-and-checkpointing)
11. [Benchmark Suite](#11-benchmark-suite)
12. [Module-Level Functions](#12-module-level-functions)

---

## 1. Core Configuration and Result Classes

### `ga.Config`
Algorithm configuration class for the genetic algorithm.

**Attributes:**
- `population_size: int` - Number of individuals in the population
- `generations: int` - Number of generations to run
- `dimension: int` - Gene vector length (dimensionality of the problem)
- `crossover_rate: float` - Crossover probability [0,1]
- `mutation_rate: float` - Per-gene mutation probability [0,1]
- `bounds: Bounds` - Gene search bounds
- `elite_ratio: float` - Elite fraction preserved each generation [0,1]
- `seed: int` - RNG seed (0 = random)

**Example:**
```python
cfg = ga.Config()
cfg.population_size = 60
cfg.generations = 200
cfg.dimension = 10
cfg.bounds = ga.Bounds(-5.12, 5.12)
```

### `ga.Bounds`
Gene value bounds specification.

**Attributes:**
- `lower: float` - Lower bound
- `upper: float` - Upper bound

**Example:**
```python
bounds = ga.Bounds(-5.0, 5.0)
```

### `ga.Result`
Results returned by `GeneticAlgorithm.run()`.

**Attributes:**
- `best_genes: list[float]` - Best gene vector found
- `best_fitness: float` - Fitness of the best individual
- `best_history: list[float]` - Best fitness per generation
- `avg_history: list[float]` - Average fitness per generation

### `ga.OptimizationResult`
Generic optimization result container.

**Attributes:**
- `best_solution: list[float]` - Best solution found
- `best_fitness: float` - Best fitness value
- `best_history: list[float]` - Best fitness history
- `avg_history: list[float]` - Average fitness history
- `pareto_objectives: list[list[float]]` - Pareto front objectives
- `pareto_genes: list[list[float]]` - Pareto front gene vectors
- `evaluations: int` - Number of evaluations performed
- `generations: int` - Number of generations executed

### `ga.Evaluation`
Evaluation/objective record for an individual.

**Attributes:**
- `objectives: list[float]` - Objective values
- `feasible: bool` - Whether the solution is feasible
- `penalty: float` - Penalty value for constraint violations

### `ga.Individual`
Generic individual container.

**Attributes:**
- `evaluation: Evaluation` - Evaluation record
- `age: int` - Age of the individual

---

## 2. Genome Representation Classes

### `ga.IGenome`
Base genome interface.

**Methods:**
- `encoding_name() -> str` - Return the encoding name

### `ga.VectorGenome`
Vector-based genome representation (most common for continuous optimization).

**Attributes:**
- `genes: list[float]` - Gene vector

**Methods:**
- `encoding_name() -> str` - Returns "vector<double>"

**Example:**
```python
genome = ga.VectorGenome([0.1, 0.2, 0.3])
print(genome.genes)  # [0.1, 0.2, 0.3]
```

### `ga.BitsetGenome`
Binary/bitset-based genome representation.

**Attributes:**
- `bits: list[bool]` - Bit vector

**Methods:**
- `size() -> int` - Number of bits
- `hamming_distance(other: BitsetGenome) -> int` - Hamming distance to another bitset
- `popcount() -> int` - Number of set bits
- `encoding_name() -> str` - Returns "bitset"

**Example:**
```python
genome = ga.BitsetGenome([True, False, True, False])
print(genome.size())  # 4
print(genome.popcount())  # 2
```

### `ga.PermutationGenome`
Permutation-based genome (for ordering/TSP problems).

**Attributes:**
- `order: list[int]` - Permutation order

**Methods:**
- `size() -> int` - Length of permutation
- `is_valid() -> bool` - Check if permutation is valid
- `position_of(value: int) -> int` - Position of a value
- `random(size: int, seed: int = 0) -> PermutationGenome` - Create random permutation
- `encoding_name() -> str` - Returns "permutation"

**Example:**
```python
genome = ga.PermutationGenome([2, 0, 1])
assert genome.is_valid()
print(genome.position_of(2))  # 0
```

### `ga.SetGenome`
Set-based genome representation.

**Attributes:**
- `values: set[int]` - Set of integer values

**Methods:**
- `encoding_name() -> str` - Returns "set<int>"

### `ga.MapGenome`
Map/dictionary-based genome representation.

**Attributes:**
- `values: dict[str, float]` - Dictionary of string keys to float values

**Methods:**
- `encoding_name() -> str` - Returns "map<string,double>"

### `ga.NdArrayGenome`
N-dimensional array genome representation.

**Attributes:**
- `rows: int` - Number of rows
- `cols: int` - Number of columns
- `data: list[float]` - Flattened data array

**Methods:**
- `get(row: int, col: int) -> float` - Get value at position
- `set(row: int, col: int, value: float)` - Set value at position
- `encoding_name() -> str` - Returns "ndarray"

**Example:**
```python
genome = ga.NdArrayGenome(2, 3)
genome.set(1, 1, 3.5)
print(genome.get(1, 1))  # 3.5
```

### `ga.TreeGenome`
Tree-based genome for genetic programming.

**Methods:**
- `has_root() -> bool` - Check if tree has a root node
- `set_root(node: Node)` - Set the root node
- `root() -> Node | None` - Get the root node
- `encoding_name() -> str` - Returns "tree"

---

## 3. Genetic Programming Classes

### `ga.ValueType` (Enum)
Type system for genetic programming.

**Values:**
- `any` - Any type
- `bool` - Boolean type
- `int` - Integer type
- `double` - Double/float type

### `ga.Signature`
Function signature for GP primitives.

**Attributes:**
- `return_type: ValueType` - Return type
- `arg_types: list[ValueType]` - Argument types

### `ga.Primitive`
GP primitive (function or terminal).

**Attributes:**
- `name: str` - Primitive name
- `signature: Signature` - Type signature
- `is_terminal: bool` - Whether this is a terminal (leaf) node

### `ga.Node`
Tree node for genetic programming.

**Attributes:**
- `symbol: str` - Node symbol/name
- `return_type: ValueType` - Return type

**Methods:**
- `size() -> int` - Total size of subtree
- `child_count() -> int` - Number of children
- `add_child(child: Node)` - Add a child node

**Example:**
```python
node = ga.Node("x", ga.ValueType.double)
child = ga.Node("const", ga.ValueType.double)
node.add_child(child)
```

### `ga.TreeBuilder`
Builder for generating GP trees.

**Constructor:**
- `TreeBuilder(primitives: list[Primitive])` - Initialize with primitive set

**Methods:**
- `grow(max_depth: int, target_type: ValueType = ValueType.any, strongly_typed: bool = False, seed: int = 0) -> Node` - Generate a tree using the grow method

### `ga.ADFPool`
Automatically Defined Function pool for GP.

**Methods:**
- `put(name: str, root: Node)` - Store an ADF
- `has(name: str) -> bool` - Check if ADF exists
- `get(name: str) -> Node` - Retrieve an ADF
- `size() -> int` - Number of ADFs stored

---

## 4. Main Algorithm Classes

### `ga.GeneticAlgorithm`
Main genetic algorithm class for single-objective optimization.

**Constructor:**
- `GeneticAlgorithm(config: Config)` - Initialize with configuration

**Methods:**
- `run(fitness: Callable[[list[float]], float]) -> Result` - Run the GA with fitness function (higher is better)
- `set_mutation_operator(op)` - Set custom mutation operator
- `set_crossover_operator(op)` - Set custom crossover operator
- `config() -> Config` - Get current configuration

**Example:**
```python
def sphere(x):
    return 1000.0 / (1.0 + sum(xi**2 for xi in x))

cfg = ga.Config()
cfg.dimension = 10
cfg.population_size = 50
cfg.generations = 100

engine = ga.GeneticAlgorithm(cfg)
result = engine.run(sphere)
print(f"Best fitness: {result.best_fitness}")
```

### `ga.Optimizer`
High-level optimizer facade with fluent interface.

**Methods:**
- `with_config(config: Config) -> Optimizer` - Set configuration
- `with_threads(threads: int) -> Optimizer` - Set thread count
- `with_seed(seed: int) -> Optimizer` - Set random seed
- `optimize(objective: Callable[[list[float]], float]) -> Result` - Run single-objective optimization
- `optimize_multi_objective_nsga2(objectives: list[Callable], population_size: int = 80, generations: int = 80) -> MultiObjectiveResult` - NSGA-II multi-objective optimization
- `optimize_multi_objective_nsga3(objectives: list[Callable], population_size: int = 80, generations: int = 80, reference_divisions: int = 8) -> MultiObjectiveResult` - NSGA-III multi-objective optimization

### `ga.OptimizerBuilder`
Fluent builder for creating optimizers.

**Methods:**
- `dimension(dimension: int) -> OptimizerBuilder` - Set problem dimension
- `bounds(lower: float, upper: float) -> OptimizerBuilder` - Set bounds
- `population_size(size: int) -> OptimizerBuilder` - Set population size
- `generations(gen: int) -> OptimizerBuilder` - Set generation count
- `seed(seed: int) -> OptimizerBuilder` - Set random seed
- `crossover_rate(rate: float) -> OptimizerBuilder` - Set crossover rate
- `mutation_rate(rate: float) -> OptimizerBuilder` - Set mutation rate
- `elite_ratio(ratio: float) -> OptimizerBuilder` - Set elite ratio
- `threads(threads: int) -> OptimizerBuilder` - Set thread count
- `build() -> Optimizer` - Build the optimizer

**Example:**
```python
opt = (ga.OptimizerBuilder()
       .dimension(5)
       .bounds(-10.0, 10.0)
       .population_size(100)
       .generations(50)
       .build())
result = opt.optimize(fitness_function)
```

### `ga.MultiObjectiveResult`
Multi-objective optimization result.

**Attributes:**
- `pareto_genes: list[list[float]]` - Gene vectors on Pareto front
- `pareto_objectives: list[list[float]]` - Objective vectors on Pareto front

---

## 5. Multi-Objective Evolutionary Algorithms (MOEA)

### NSGA-II

#### `ga.Nsga2Config`
NSGA-II configuration.

**Attributes:**
- `population_size: int` - Population size
- `generations: int` - Generation count
- `seed: int` - Random seed

#### `ga.Nsga2`
NSGA-II utility methods for objective-space operations.

**Constructor:**
- `Nsga2(config: Nsga2Config = Nsga2Config())` - Initialize NSGA-II

**Methods:**
- `non_dominated_sort_objectives(objectives: list[list[float]]) -> list[list[int]]` - Non-dominated sorting (assumes minimization)
- `crowding_distance_objectives(objectives: list[list[float]], front: list[int]) -> list[float]` - Compute crowding distance

**Example:**
```python
objectives = [[0.5, 1.0], [1.0, 0.5], [0.8, 0.8]]
nsga2 = ga.Nsga2()
fronts = nsga2.non_dominated_sort_objectives(objectives)
distances = nsga2.crowding_distance_objectives(objectives, fronts[0])
```

### NSGA-III

#### `ga.Nsga3`
NSGA-III utility methods for objective-space operations.

**Constructor:**
- `Nsga3(config: Nsga2Config = Nsga2Config())` - Initialize NSGA-III

**Static Methods:**
- `generate_reference_points(objective_count: int, divisions: int) -> list[list[float]]` - Generate Das-Dennis reference points

**Methods:**
- `non_dominated_sort_objectives(objectives: list[list[float]]) -> list[list[int]]` - Non-dominated sorting
- `environmental_select_objectives(objectives: list[list[float]], target_size: int, reference_points: list[list[float]]) -> list[list[float]]` - Select next generation
- `environmental_select_indices(objectives: list[list[float]], target_size: int, reference_points: list[list[float]]) -> list[int]` - Select indices

**Example:**
```python
refs = ga.Nsga3.generate_reference_points(2, 4)
nsga3 = ga.Nsga3()
selected = nsga3.environmental_select_indices(objectives, 10, refs)
```

### SPEA2

#### `ga.Spea2`
SPEA2 objective-space utilities.

**Methods:**
- `strength_fitness_objectives(objectives: list[list[float]]) -> list[float]` - Compute SPEA2 strength fitness (lower is better)
- `environmental_select_objectives(objectives: list[list[float]], target_size: int) -> list[list[float]]` - Select next generation
- `environmental_select_indices(objectives: list[list[float]], target_size: int) -> list[int]` - Select indices

---

## 6. Evolution Strategies (ES/CMA-ES/MO-CMA-ES)

### Evolution Strategy (ES)

#### `ga.EvolutionStrategyConfig`
Evolution strategy configuration.

**Attributes:**
- `mu: int` - Number of parents
- `lambda_: int` - Number of offspring (use `lambda_` in Python due to keyword)
- `generations: int` - Generation count
- `dimension: int` - Problem dimension
- `sigma: float` - Step size
- `lower: float` - Lower bound
- `upper: float` - Upper bound
- `plus_strategy: bool` - Whether to use plus strategy (mu+lambda) vs comma strategy (mu,lambda)
- `seed: int` - Random seed

#### `ga.EvolutionStrategyResult`
Evolution strategy result.

**Attributes:**
- `best: list[float]` - Best solution
- `best_fitness: float` - Best fitness value
- `best_history: list[float]` - Fitness history

#### `ga.EvolutionStrategy`
Evolution strategy optimizer.

**Constructor:**
- `EvolutionStrategy(config: EvolutionStrategyConfig)` - Initialize ES

**Methods:**
- `run(fitness: Callable[[list[float]], float]) -> EvolutionStrategyResult` - Run ES optimization

### CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

#### `ga.CmaEsConfig`
CMA-ES configuration.

**Attributes:**
- `population_size: int` - Population size
- `generations: int` - Generation count
- `dimension: int` - Problem dimension
- `lower: float` - Lower bound
- `upper: float` - Upper bound
- `sigma: float` - Initial step size
- `seed: int` - Random seed

#### `ga.CmaEsResult`
CMA-ES result.

**Attributes:**
- `best: list[float]` - Best solution
- `best_fitness: float` - Best fitness value
- `history: list[float]` - Fitness history

#### `ga.DiagonalCmaEs`
Diagonal CMA-ES optimizer (memory-efficient variant).

**Constructor:**
- `DiagonalCmaEs(config: CmaEsConfig)` - Initialize CMA-ES

**Methods:**
- `run(fitness: Callable[[list[float]], float]) -> CmaEsResult` - Run CMA-ES optimization

**Example:**
```python
cma_cfg = ga.CmaEsConfig()
cma_cfg.dimension = 10
cma_cfg.population_size = 20
cma_cfg.generations = 100
cma = ga.DiagonalCmaEs(cma_cfg)
result = cma.run(fitness_function)
```

### MO-CMA-ES (Multi-Objective CMA-ES)

#### `ga.MoCmaEsConfig`
MO-CMA-ES configuration.

**Attributes:**
- `cma: CmaEsConfig` - Base CMA-ES config
- `weights: list[float]` - Objective weights

#### `ga.MoCmaEsResult`
MO-CMA-ES result.

**Attributes:**
- `best: list[float]` - Best solution
- `objectives: list[float]` - Objective values
- `weighted_fitness: float` - Weighted fitness value

#### `ga.MoCmaEs`
Multi-objective CMA-ES optimizer.

**Constructor:**
- `MoCmaEs(config: MoCmaEsConfig)` - Initialize MO-CMA-ES

**Methods:**
- `run(objective: Callable[[list[float]], list[float]]) -> MoCmaEsResult` - Run MO-CMA-ES

---

## 7. Constraints and Adaptive Control

### `ga.ConstraintSet`
Container for constraint functions.

**Methods:**
- `add_hard_constraint(constraint: Callable[[list[float]], bool])` - Add hard constraint (must be satisfied)
- `add_soft_penalty(penalty: Callable[[list[float]], float])` - Add soft penalty function
- `add_repair(repair: Callable[[list[float]], list[float]])` - Add repair function
- `clear()` - Clear all constraints

**Example:**
```python
cs = ga.ConstraintSet()
cs.add_hard_constraint(lambda x: all(-1.0 <= v <= 1.0 for v in x))
cs.add_soft_penalty(lambda x: sum(max(0, abs(v) - 1.0) for v in x))
```

### `ga.AdaptiveRates`
Adaptive parameter rates.

**Attributes:**
- `mutation_rate: float` - Current mutation rate
- `crossover_rate: float` - Current crossover rate

### `ga.AdaptiveRateController`
Controller for adaptive rate adjustment.

**Constructor:**
- `AdaptiveRateController(min_mutation: float = 0.001, max_mutation: float = 0.6, min_crossover: float = 0.4, max_crossover: float = 0.95)` - Initialize controller

**Methods:**
- `update(current: AdaptiveRates, diversity: float, best_improvement: float) -> AdaptiveRates` - Update rates based on diversity and improvement

---

## 8. Hybrid and Coevolution

### `ga.HybridOptimizer`
Hybrid optimizer combining GA with local search.

**Constructor:**
- `HybridOptimizer(config: Config)` - Initialize hybrid optimizer

**Methods:**
- `run(fitness: Callable, local_search: Callable = None, local_search_restarts: int = 5) -> Result` - Run hybrid optimization

**Example:**
```python
hybrid = ga.HybridOptimizer(cfg)
result = hybrid.run(fitness, lambda x: [0.9 * v for v in x], 3)
```

### `ga.CoevolutionConfig`
Coevolution configuration.

**Attributes:**
- `generations: int` - Generation count
- `seed: int` - Random seed

### `ga.CoevolutionEngine`
Coevolution engine for competitive/cooperative evolution.

**Constructor:**
- `CoevolutionEngine(config: CoevolutionConfig)` - Initialize coevolution engine

**Methods:**
- `run(populations: list[list[Individual]], evaluate: Callable = None, reproduce: Callable = None) -> list[list[Individual]]` - Run coevolution

---

## 9. Parallel Evaluation

### `ga.ParallelEvaluator`
Threaded batch evaluator for parallel fitness evaluation.

**Constructor:**
- `ParallelEvaluator(fitness: Callable[[list[float]], float], threads: int = hardware_concurrency)` - Initialize parallel evaluator

**Methods:**
- `evaluate(batch: list[list[float]]) -> list[float]` - Evaluate a batch in parallel

**Example:**
```python
evaluator = ga.ParallelEvaluator(fitness_function, threads=4)
batch = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
results = evaluator.evaluate(batch)
```

### `ga.LocalDistributedExecutor`
Local threaded distributed executor for fitness evaluation.

**Constructor:**
- `LocalDistributedExecutor(evaluator: Callable[[list[float]], float], workers: int = hardware_concurrency)` - Initialize executor

**Methods:**
- `execute(batch: list[list[float]]) -> list[float]` - Execute batch evaluation

---

## 10. Tracking, Visualization, and Checkpointing

### `ga.ExperimentTracker`
Experiment tracking and logging utility.

**Constructor:**
- `ExperimentTracker(run_id: str)` - Initialize tracker with run ID

**Methods:**
- `write_config(config: Config, path: str)` - Write configuration to file
- `write_history_csv(result: Result, path: str)` - Write fitness history to CSV
- `write_best_solution_csv(result: Result, path: str)` - Write best solution to CSV

### `ga.CheckpointState`
Checkpoint serialization state.

**Attributes:**
- `config: Config` - Algorithm configuration
- `result: Result` - Current results
- `generation: int` - Current generation
- `rng_state: str` - RNG state string

**Example:**
```python
state = ga.CheckpointState()
state.config = cfg
state.result = result
state.generation = 100
state.rng_state = "checkpoint-100"

ga.checkpoint_save_binary("checkpoint.bin", state)
loaded = ga.checkpoint_load_binary("checkpoint.bin")
```

---

## 11. Benchmark Suite

### `ga.BenchmarkConfig`
Benchmark configuration.

**Attributes:**
- `warmup_iterations: int` - Number of warmup iterations
- `benchmark_iterations: int` - Number of benchmark iterations
- `verbose: bool` - Verbose output
- `csv_output: bool` - Enable CSV output
- `output_file: str` - Output file path

### `ga.BenchmarkResult`
Scalability benchmark aggregate result.

**Attributes (readonly):**
- `name: str` - Benchmark name
- `category: str` - Benchmark category
- `avg_execution_time: float` - Average execution time
- `min_execution_time: float` - Minimum execution time
- `max_execution_time: float` - Maximum execution time
- `iterations: int` - Number of iterations
- `throughput: float` - Operations per second
- `standard_deviation: float` - Standard deviation
- `success: bool` - Whether benchmark succeeded
- `error_message: str` - Error message if failed

### `ga.OperatorBenchmark`
Operator-level benchmark result.

**Attributes (readonly):**
- `operator_name: str` - Operator name
- `operator_type: str` - Operator type
- `avg_time: float` - Average time
- `operations_per_second: float` - Operations per second
- `iterations: int` - Number of iterations
- `representation: str` - Genome representation

### `ga.FunctionBenchmark`
Function optimization benchmark result.

**Attributes (readonly):**
- `function_name: str` - Function name
- `best_fitness: float` - Best fitness achieved
- `avg_fitness: float` - Average fitness
- `generations_to_converge: int` - Generations to convergence
- `total_execution_time: float` - Total execution time
- `best_solution: list[float]` - Best solution found
- `convergence_history: list[float]` - Convergence history

### `ga.GABenchmark`
Benchmark suite runner.

**Constructor:**
- `GABenchmark(config: BenchmarkConfig = BenchmarkConfig())` - Initialize benchmark suite

**Methods:**
- `run_all_benchmarks()` - Run all benchmarks
- `run_operator_benchmarks()` - Run operator benchmarks only
- `run_function_benchmarks()` - Run function benchmarks only
- `run_scalability_benchmarks()` - Run scalability benchmarks only
- `generate_report()` - Generate benchmark report
- `export_to_csv(filename: str)` - Export results to CSV
- `operator_results() -> list[OperatorBenchmark]` - Get operator benchmark results
- `function_results() -> list[FunctionBenchmark]` - Get function benchmark results
- `scalability_results() -> list[BenchmarkResult]` - Get scalability benchmark results

**Example:**
```python
cfg = ga.BenchmarkConfig()
cfg.benchmark_iterations = 10
cfg.verbose = True
bench = ga.GABenchmark(cfg)
bench.run_operator_benchmarks()
bench.export_to_csv("benchmark_results.csv")
```

---

## 12. Module-Level Functions

### Operator Factories

#### `ga.make_gaussian_mutation(seed: int = 0)`
Create a Gaussian mutation operator.

#### `ga.make_uniform_mutation(seed: int = 0)`
Create a Uniform mutation operator.

#### `ga.make_one_point_crossover(seed: int = 0)`
Create a One-Point crossover operator.

#### `ga.make_two_point_crossover(seed: int = 0)`
Create a Two-Point crossover operator.

**Example:**
```python
mutation = ga.make_gaussian_mutation(seed=42)
crossover = ga.make_one_point_crossover(seed=42)

engine = ga.GeneticAlgorithm(cfg)
engine.set_mutation_operator(mutation)
engine.set_crossover_operator(crossover)
```

### Selection Helpers

#### `ga.selection_tournament_indices(fitness: list[float], tournament_size: int = 3) -> list[int]`
Tournament selection - returns one winner index from the tournament.

#### `ga.selection_roulette_indices(fitness: list[float], count: int) -> list[int]`
Roulette-wheel selection - returns selected indices.

#### `ga.selection_rank_indices(fitness: list[float], count: int) -> list[int]`
Rank selection - returns selected indices.

#### `ga.selection_sus_indices(fitness: list[float], count: int) -> list[int]`
Stochastic universal sampling - returns selected indices.

#### `ga.selection_elitism_indices(fitness: list[float], elite_count: int) -> list[int]`
Elitism helper - returns indices of top-fitness individuals.

**Example:**
```python
fitness = [0.1, 0.8, 0.4, 1.2, 0.6]
winner = ga.selection_tournament_indices(fitness, tournament_size=3)
elite = ga.selection_elitism_indices(fitness, elite_count=2)
```

### NSGA-II Functions

#### `ga.nsga2_non_dominated_sort(objectives: list[list[float]]) -> list[list[int]]`
Convenience function: non-dominated sorting in objective space (assumes minimization).

#### `ga.nsga2_crowding_distance(objectives: list[list[float]], front: list[int]) -> list[float]`
Convenience function: crowding distance in objective space.

### NSGA-III Functions

#### `ga.nsga3_reference_points(objective_count: int, divisions: int) -> list[list[float]]`
Generate Das-Dennis reference points for NSGA-III.

#### `ga.nsga3_environmental_select_indices(objectives: list[list[float]], target_size: int, reference_points: list[list[float]]) -> list[int]`
NSGA-III environmental selection over objective vectors.

### SPEA2 Functions

#### `ga.spea2_strength_fitness(objectives: list[list[float]]) -> list[float]`
SPEA2 strength fitness in objective space (lower is better).

#### `ga.spea2_environmental_select_indices(objectives: list[list[float]], target_size: int) -> list[int]`
SPEA2 environmental selection indices.

### Constraint Functions

#### `ga.is_feasible(genes: list[float], constraint_set: ConstraintSet) -> bool`
Check if genes satisfy all hard constraints.

#### `ga.total_penalty(genes: list[float], constraint_set: ConstraintSet) -> float`
Calculate total penalty from soft constraints.

#### `ga.apply_repairs(genes: list[float], constraint_set: ConstraintSet) -> list[float]`
Apply repair functions to genes.

#### `ga.penalized_fitness(base_fitness: float, genes: list[float], constraint_set: ConstraintSet, infeasible_penalty: float = 1e6) -> float`
Calculate penalized fitness considering constraints.

### Checkpoint Functions

#### `ga.checkpoint_save_json(path: str, state: CheckpointState)`
Save checkpoint state as JSON.

#### `ga.checkpoint_load_json(path: str) -> CheckpointState`
Load checkpoint state from JSON.

#### `ga.checkpoint_save_binary(path: str, state: CheckpointState)`
Save checkpoint state as binary.

#### `ga.checkpoint_load_binary(path: str) -> CheckpointState`
Load checkpoint state from binary.

**Example:**
```python
# Save checkpoint
state = ga.CheckpointState()
state.config = cfg
state.result = result
state.generation = 50
ga.checkpoint_save_json("checkpoint.json", state)

# Load checkpoint
loaded_state = ga.checkpoint_load_json("checkpoint.json")
print(f"Loaded generation: {loaded_state.generation}")
```

### Visualization/Export Functions

#### `ga.export_fitness_curve_csv(best: list[float], avg: list[float], path: str)`
Export fitness curves to CSV.

#### `ga.export_pareto_front_csv(objectives: list[list[float]], path: str)`
Export Pareto front to CSV.

#### `ga.export_diversity_csv(diversity: list[float], path: str)`
Export diversity metrics to CSV.

**Example:**
```python
result = engine.run(fitness)
ga.export_fitness_curve_csv(result.best_history, result.avg_history, "fitness_curve.csv")
```

---

## Summary Statistics

- **Total Classes:** 52
- **Total Functions:** 26
- **API Categories:** 12

### Class Distribution by Category

1. Core Configuration: 6 classes
2. Genome Representations: 7 classes
3. Genetic Programming: 6 classes
4. Main Algorithms: 4 classes
5. Multi-Objective EA: 6 classes
6. Evolution Strategies: 9 classes
7. Constraints/Adaptive: 3 classes
8. Hybrid/Coevolution: 3 classes
9. Parallel Evaluation: 2 classes
10. Tracking/Checkpointing: 2 classes
11. Benchmark Suite: 4 classes

### All APIs Verified

✓ All expected APIs are exposed and accessible through the Python bindings.

---

## Installation

### Build from source:
```bash
python3 -m pip install pybind11
cmake -S . -B build
cmake --build build --target ga_python_module
```

### Install with pip:
```bash
pip install .
```

Or for development:
```bash
pip install -e .
```

---

## Quick Start Example

```python
import ga

# Configure the algorithm
cfg = ga.Config()
cfg.population_size = 50
cfg.generations = 100
cfg.dimension = 10
cfg.bounds = ga.Bounds(-5.0, 5.0)
cfg.mutation_rate = 0.05
cfg.crossover_rate = 0.8

# Define fitness function (higher is better)
def sphere(x):
    return 1000.0 / (1.0 + sum(xi**2 for xi in x))

# Run optimization
engine = ga.GeneticAlgorithm(cfg)
result = engine.run(sphere)

print(f"Best fitness: {result.best_fitness}")
print(f"Best solution: {result.best_genes}")

# Export results
ga.export_fitness_curve_csv(result.best_history, result.avg_history, "fitness.csv")
```

---

**Note:** This document is automatically verified against the actual Python bindings. All listed APIs have been tested for existence and basic functionality.
