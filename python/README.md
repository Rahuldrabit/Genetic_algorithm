# Python Bindings Guide

This repository provides a Python package named `genetic_algorithm_lib` backed by a compiled
extension module (`_core`) built from `python/ga_bindings.cpp`.

## Build

From the repository root:

```bash
python3 -m pip install --user pybind11
cmake -S . -B build
cmake --build build --target ga_python_module
```

The module is produced under `build/python`.

## Install with pip

From the repository root:

```bash
python3 -m pip install .
```

Or for editable/development install:

```bash
python3 -m pip install -e .
```

Then import directly:

```python
import genetic_algorithm_lib as ga
```

## Example

Run the bundled example:

```bash
python3 python/example.py
```

## NSGA-II Objective-Space Utilities

- `ga.Nsga2`
- `ga.nsga2_non_dominated_sort(objectives)`
- `ga.nsga2_crowding_distance(objectives, front)`

`objectives` is a list of objective vectors (`list[list[float]]`) and assumes minimization.

## NSGA-III Objective-Space Utilities

- `ga.Nsga3`
- `ga.Nsga3.generate_reference_points(objective_count, divisions)`
- `ga.nsga3_reference_points(objective_count, divisions)`
- `ga.Nsga3.environmental_select_objectives(objectives, target_size, reference_points)`
- `ga.Nsga3.environmental_select_indices(objectives, target_size, reference_points)`
- `ga.nsga3_environmental_select_indices(objectives, target_size, reference_points)`

Use `environmental_select_indices` when you want selected indices back into your original objective list.

## Additional Python-Exposed Features

The bindings now also expose these public framework APIs:

- Core/representations: `Evaluation`, `Individual`, `IGenome`, `VectorGenome`, `BitsetGenome`, `PermutationGenome`, `SetGenome`, `MapGenome`, `NdArrayGenome`, `TreeGenome`
- Operators + plugin registries: `CrossoverOperator`, `MutationOperator`, concrete crossover/mutation operators (e.g. `BlendCrossover`, `OrderCrossover`, `BitFlipMutation`, ...), plus `CrossoverRegistry` and `MutationRegistry`
- GP helpers: `ValueType`, `Signature`, `Primitive`, `Node`, `TreeBuilder`, `ADFPool`
- Optimizer APIs: `Optimizer`, `OptimizerBuilder`, `MultiObjectiveResult`, `OptimizationResult`
- Advanced algorithms: `Spea2`, `EvolutionStrategy`, `DiagonalCmaEs`, `MoCmaEs`
- Supporting utilities: `ConstraintSet`, `AdaptiveRates`, `AdaptiveRateController`, `HybridOptimizer`, `CoevolutionEngine`, `ExperimentTracker`
- Export/checkpoint helpers:
  - `checkpoint_save_binary(path, state)`, `checkpoint_load_binary(path)`
  - `export_fitness_curve_csv(best, avg, path)`
  - `export_pareto_front_csv(objectives, path)`
  - `export_diversity_csv(diversity, path)`

## Checkpoint JSON API

- `ga.CheckpointState`
- `ga.checkpoint_save_json(path, state)`
- `ga.checkpoint_load_json(path)`

Minimal usage:

```python
import genetic_algorithm_lib as ga

cfg = ga.Config()
res = ga.Result()

state = ga.CheckpointState()
state.config = cfg
state.result = res
state.generation = 0
state.rng_state = "seed-42"

ga.checkpoint_save_json("checkpoint.json", state)
loaded = ga.checkpoint_load_json("checkpoint.json")
print(loaded.generation)
```
