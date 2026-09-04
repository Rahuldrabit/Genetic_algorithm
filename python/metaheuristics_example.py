"""Run user-configured C++ metaheuristics from Python."""

from __future__ import annotations

import genetic_algorithm_lib as ga


def sphere_fitness(solution: list[float]) -> float:
    return 1.0 / (1.0 + sum(value * value for value in solution))


def search_config(seed: int) -> ga.SearchConfig:
    config = ga.SearchConfig()
    config.population_size = 40
    config.iterations = 50
    config.dimension = 5
    config.bounds = ga.Bounds(-5.0, 5.0)
    config.seed = seed
    config.threads = 1
    return config


def main() -> None:
    # Optional fuzzy control. Leave each optimizer's controller as None when
    # fixed, user-supplied algorithm parameters are desired.
    fuzzy_config = ga.FuzzyControllerConfig()
    fuzzy_config.improvement_scale = 20.0
    fuzzy_config.low_diversity_stagnant = ga.ControlSignal(1.55, 0.80, 1.30, 1.45)
    fuzzy = ga.FuzzyAdaptiveController(fuzzy_config)

    ga_config = ga.Config()
    ga_config.population_size = 40
    ga_config.generations = 50
    ga_config.dimension = 5
    ga_config.bounds = ga.Bounds(-5.0, 5.0)
    ga_config.seed = 42

    pso_config = ga.PsoConfig()
    pso_config.search = search_config(43)
    pso_config.variant = ga.PsoVariant.constriction
    pso_config.inertia = 0.7298
    pso_config.cognitive = 1.49618
    pso_config.social = 1.49618
    pso_config.controller = fuzzy

    acor_config = ga.AcorConfig()
    acor_config.search = search_config(44)
    acor_config.archive_size = 40
    acor_config.sample_count = 20
    acor_config.locality = 0.5
    acor_config.convergence_speed = 0.85
    acor_config.controller = fuzzy

    # Only these explicitly added stages run, in precisely this order.
    pipeline = ga.MetaheuristicPipeline()
    pipeline.add(ga.GeneticAlgorithmAdapter(ga_config))
    pipeline.add(ga.ParticleSwarmOptimizer(pso_config))
    pipeline.add(ga.ContinuousAntColonyOptimizer(acor_config))

    result = pipeline.optimize_detailed(sphere_fitness)
    print("Best fitness:", result.combined.best_fitness)
    print("Evaluations:", result.combined.evaluations)
    for stage in result.stages:
        print(stage.optimizer, stage.result.best_fitness)


if __name__ == "__main__":
    main()
