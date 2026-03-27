"""
Minimal Python example for the Genetic Algorithm framework.

Optimises the Rastrigin function (minimisation → maximisation via 1000/(1+f)).
Global minimum is at origin where f=0, giving fitness ≈ 1000.
"""
import math
import sys
import os

# Add the build directory to the path so Python can find the 'ga' module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'python'))

import ga  # C++ extension module


def rastrigin_fitness(x: list[float]) -> float:
    """Rastrigin function converted to a maximisation objective."""
    A = 10.0
    val = A * len(x) + sum(xi**2 - A * math.cos(2 * math.pi * xi) for xi in x)
    return 1000.0 / (1.0 + val)


def main():
    cfg = ga.Config()
    cfg.population_size = 60
    cfg.generations = 200
    cfg.dimension = 10
    cfg.bounds = ga.Bounds(-5.12, 5.12)
    cfg.mutation_rate = 0.05
    cfg.crossover_rate = 0.8
    cfg.elite_ratio = 0.05

    engine = ga.GeneticAlgorithm(cfg)

    print("Running Genetic Algorithm on the Rastrigin function...")
    result = engine.run(rastrigin_fitness)

    print(f"Best fitness : {result.best_fitness:.4f}")
    print(f"Best genes   : {[f'{g:.4f}' for g in result.best_genes]}")
    print(f"Generations  : {len(result.best_history)}")

    # Show convergence trend (every 50 generations)
    print("\nConvergence (best fitness per 50 generations):")
    for i, f in enumerate(result.best_history):
        if i % 50 == 0 or i == len(result.best_history) - 1:
            print(f"  gen {i:3d}: {f:.4f}")

    # NSGA-III utility demo in objective space
    objectives = [
        [0.02, 1.20],
        [0.10, 0.80],
        [0.35, 0.45],
        [0.90, 0.10],
    ]
    refs = ga.nsga3_reference_points(2, 4)
    picked = ga.nsga3_environmental_select_indices(objectives, 2, refs)
    print(f"\nNSGA-III selected objective indices: {picked}")

    # Checkpoint JSON demo
    state = ga.CheckpointState()
    state.config = cfg
    state.result = result
    state.generation = len(result.best_history) - 1
    state.rng_state = "python-demo"
    cp_path = os.path.join(os.path.dirname(__file__), "..", "build", "python_checkpoint.json")
    ga.checkpoint_save_json(cp_path, state)
    loaded = ga.checkpoint_load_json(cp_path)
    print(f"Checkpoint JSON roundtrip generation: {loaded.generation}")


if __name__ == "__main__":
    main()
