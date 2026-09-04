"""Sanity checks for Python bindings coverage."""

from __future__ import annotations

import os
import sys

try:
    import genetic_algorithm_lib as ga
except ImportError:
    # Allow running from the source tree after a local CMake build.
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    sys.path.insert(0, os.path.join(repo_root, "python"))
    module_dir = os.environ.get(
        "GA_PYTHON_MODULE_DIR", os.path.join(repo_root, "build", "python")
    )
    sys.path.insert(0, module_dir)
    import genetic_algorithm_lib as ga


def sphere_fitness(x: list[float]) -> float:
    return 1000.0 / (1.0 + sum(v * v for v in x))


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "build")
    os.makedirs(out_dir, exist_ok=True)

    # Core data/representations
    ev = ga.Evaluation()
    ev.objectives = [1.0, 2.0]
    ind = ga.Individual()
    ind.evaluation = ev
    ind.age = 1

    assert ga.VectorGenome([0.1, 0.2]).encoding_name() == "vector<double>"
    assert ga.VectorGenomeInt([1, 2, 3]).encoding_name() == "vector<int>"
    assert ga.BitsetGenome([True, False]).size() == 2
    assert ga.PermutationGenome([2, 0, 1]).is_valid()
    assert ga.SetGenome({1, 2}).encoding_name() == "set<int>"
    assert ga.MapGenome({"x": 1.0}).encoding_name() == "map<string,double>"

    arr = ga.NdArrayGenome(2, 2)
    arr.set(1, 1, 3.0)
    assert arr.get(1, 1) == 3.0

    node = ga.Node("x", ga.ValueType.double)
    node.add_child(ga.Node("c", ga.ValueType.double))
    tree = ga.TreeGenome()
    tree.set_root(node)
    assert tree.has_root()

    # Optimizer + advanced algorithms
    opt = (
        ga.OptimizerBuilder()
        .dimension(3)
        .bounds(-1.0, 1.0)
        .population_size(12)
        .generations(6)
        .seed(42)
        .build()
    )
    res = opt.optimize(sphere_fitness)
    assert len(res.best_genes) == 3

    # User-configured metaheuristics
    search = ga.SearchConfig()
    search.population_size = 12
    search.iterations = 5
    search.dimension = 3
    search.bounds = ga.Bounds(-2.0, 2.0)
    search.seed = 31
    search.threads = 2

    fuzzy_cfg = ga.FuzzyControllerConfig()
    fuzzy_cfg.improvement_scale = 12.0
    fuzzy_cfg.low_diversity_stagnant = ga.ControlSignal(1.8, 0.8, 1.2, 1.5)
    fuzzy = ga.FuzzyAdaptiveController(fuzzy_cfg)
    state = ga.ProgressState()
    state.stagnation = 1.0
    assert fuzzy.update(state).exploration > 1.0

    class FixedController(ga.AdaptiveController):
        def update(self, _state):
            return ga.ControlSignal(1.2, 1.0, 1.0, 1.0)

    custom_controller_cfg = ga.PsoConfig()
    custom_controller_cfg.search = search
    # Deliberately assign a temporary object: the C++ config must retain the
    # Python override for the optimizer's full lifetime.
    custom_controller_cfg.controller = FixedController()
    custom_controller_result = ga.ParticleSwarmOptimizer(
        custom_controller_cfg
    ).optimize(sphere_fitness)
    assert len(custom_controller_result.best_solution) == 3

    pso_variants = [
        ga.PsoVariant.global_best,
        ga.PsoVariant.local_best,
        ga.PsoVariant.constriction,
        ga.PsoVariant.bare_bones,
        ga.PsoVariant.fully_informed,
        ga.PsoVariant.quantum_behaved,
        ga.PsoVariant.binary,
    ]
    for index, variant in enumerate(pso_variants):
        pso_cfg = ga.PsoConfig()
        pso_cfg.search = search
        pso_cfg.search.seed = 40 + index
        pso_cfg.variant = variant
        pso_cfg.controller = fuzzy
        if variant == ga.PsoVariant.binary:
            pso_cfg.search.bounds = ga.Bounds(0.0, 1.0)
        pso_result = ga.ParticleSwarmOptimizer(pso_cfg).optimize(sphere_fitness)
        assert len(pso_result.best_solution) == 3

    graph = ga.DenseGraph([
        [0.0, 2.0, 9.0, 10.0],
        [2.0, 0.0, 6.0, 4.0],
        [9.0, 6.0, 0.0, 8.0],
        [10.0, 4.0, 8.0, 0.0],
    ])
    aco_variants = [
        ga.AntColonyVariant.ant_system,
        ga.AntColonyVariant.elitist_ant_system,
        ga.AntColonyVariant.rank_based_ant_system,
        ga.AntColonyVariant.ant_colony_system,
        ga.AntColonyVariant.max_min_ant_system,
    ]
    for index, variant in enumerate(aco_variants):
        aco_cfg = ga.AntColonyConfig()
        aco_cfg.ants = 8
        aco_cfg.iterations = 5
        aco_cfg.seed = 60 + index
        aco_cfg.variant = variant
        aco_cfg.controller = fuzzy
        tour = ga.AntColonyOptimizer(aco_cfg).solve(graph)
        assert len(tour.best_tour) == graph.size

    acor_cfg = ga.AcorConfig()
    acor_cfg.search = search
    acor_cfg.archive_size = 12
    acor_cfg.sample_count = 8
    acor_cfg.controller = fuzzy
    acor = ga.ContinuousAntColonyOptimizer(acor_cfg)
    assert len(acor.optimize(sphere_fitness).best_solution) == 3

    gsa_cfg = ga.GsaConfig()
    gsa_cfg.search = search
    gsa_cfg.controller = fuzzy
    gsa = ga.GravitationalSearchOptimizer(gsa_cfg)
    assert len(gsa.optimize(sphere_fitness).best_solution) == 3

    fcm_cfg = ga.FuzzyCMeansConfig()
    fcm_cfg.clusters = 2
    fcm_cfg.seed = 73
    clusters = ga.FuzzyCMeans(fcm_cfg).fit([
        [0.0, 0.1], [0.1, 0.0], [-0.1, 0.0],
        [9.9, 10.0], [10.0, 9.9], [10.1, 10.0],
    ])
    assert len(clusters.centers) == 2
    assert len(clusters.labels()) == 6

    # The user alone selects the hybrid members and their exact order.
    ga_cfg = ga.Config()
    ga_cfg.population_size = 12
    ga_cfg.generations = 5
    ga_cfg.dimension = 3
    ga_cfg.bounds = ga.Bounds(-2.0, 2.0)
    ga_cfg.seed = 81
    pipeline_pso_cfg = ga.PsoConfig()
    pipeline_pso_cfg.search = search
    pipeline_pso_cfg.variant = ga.PsoVariant.constriction
    pipeline = ga.MetaheuristicPipeline()
    pipeline.add(ga.GeneticAlgorithmAdapter(ga_cfg))
    pipeline.add(ga.ParticleSwarmOptimizer(pipeline_pso_cfg))
    pipeline.add(ga.ContinuousAntColonyOptimizer(acor_cfg))
    hybrid_result = pipeline.optimize_detailed(sphere_fitness)
    assert [stage.optimizer for stage in hybrid_result.stages] == [
        "GA", "PSO-constriction", "ACOR"
    ]
    assert len(hybrid_result.combined.best_solution) == 3

    class PythonStage(ga.ContinuousOptimizer):
        def name(self):
            return "python-stage"

        def optimize(self, fitness, seeds):
            stage_result = ga.OptimizationResult()
            stage_result.best_solution = list(seeds[0]) if seeds else [0.0] * 3
            stage_result.best_fitness = fitness(stage_result.best_solution)
            stage_result.best_history = [stage_result.best_fitness]
            stage_result.avg_history = [stage_result.best_fitness]
            stage_result.evaluations = 1
            stage_result.generations = 1
            return stage_result

    extension_pipeline = ga.MetaheuristicPipeline()
    extension_pipeline.add(PythonStage())
    extension_result = extension_pipeline.optimize_detailed(sphere_fitness)
    assert extension_result.stages[0].optimizer == "python-stage"

    mo = opt.optimize_multi_objective_nsga2(
        [lambda x: x[0] * x[0], lambda x: (x[0] - 1.0) * (x[0] - 1.0)],
        population_size=16,
        generations=4,
    )
    assert isinstance(mo.pareto_objectives, list)

    escfg = ga.EvolutionStrategyConfig()
    escfg.dimension = 3
    escfg.generations = 4
    escfg.mu = 3
    escfg.lambda_ = 8
    esr = ga.EvolutionStrategy(escfg).run(sphere_fitness)
    assert esr.best_fitness > 0

    cma = ga.CmaEsConfig()
    cma.dimension = 3
    cma.population_size = 10
    cma.generations = 4
    cmr = ga.DiagonalCmaEs(cma).run(sphere_fitness)
    assert cmr.best_fitness > 0

    mocfg = ga.MoCmaEsConfig()
    mocfg.cma = cma
    mocfg.weights = [0.5, 0.5]
    mres = ga.MoCmaEs(mocfg).run(lambda x: [x[0] * x[0], (x[0] - 1.0) * (x[0] - 1.0)])
    assert len(mres.objectives) == 2

    # SPEA2 objective-space utilities
    objectives = [[0.0, 1.0], [0.2, 0.8], [0.8, 0.2], [1.0, 0.0]]
    assert len(ga.spea2_strength_fitness(objectives)) == len(objectives)
    assert len(ga.spea2_environmental_select_indices(objectives, 2)) == 2

    # Adaptive + constraints helpers
    rates = ga.AdaptiveRates()
    updated = ga.AdaptiveRateController().update(rates, 0.1, 0.0)
    assert updated.mutation_rate >= rates.mutation_rate

    cs = ga.ConstraintSet()
    cs.add_hard_constraint(lambda x: all(-1.0 <= v <= 1.0 for v in x))
    assert ga.is_feasible([0.5, -0.2], cs)
    assert not ga.is_feasible([2.0], cs)

    # Evaluation helpers
    pe = ga.ParallelEvaluator(sphere_fitness, threads=2)
    pe_results = pe.evaluate([[0.1, 0.2], [0.0, 0.0], [0.6, 0.7]])
    assert len(pe_results) == 3 and pe_results[1] >= pe_results[0]

    local_exec = ga.LocalDistributedExecutor(sphere_fitness, workers=2)
    local_results = local_exec.execute([[0.2, 0.1], [0.4, 0.5]])
    assert len(local_results) == 2 and all(r > 0.0 for r in local_results)

    # Selection helpers
    fitness = [0.1, 0.8, 0.4, 1.2, 0.6]
    t_idx = ga.selection_tournament_indices(fitness, tournament_size=3)
    assert len(t_idx) == 1 and 0 <= t_idx[0] < len(fitness)
    rw_idx = ga.selection_roulette_indices(fitness, count=3)
    assert len(rw_idx) == 3 and all(0 <= i < len(fitness) for i in rw_idx)
    rank_idx = ga.selection_rank_indices(fitness, count=3)
    # Legacy rank helper can return fewer indices than requested in this codebase.
    assert len(rank_idx) <= 3
    sus_idx = ga.selection_sus_indices(fitness, count=3)
    # Legacy SUS helper can return fewer indices than requested in this codebase.
    assert len(sus_idx) <= 3
    elite_idx = ga.selection_elitism_indices(fitness, elite_count=2)
    assert len(elite_idx) == 2 and all(0 <= i < len(fitness) for i in elite_idx)

    # Operator classes + plugin registries
    p1 = [0.1, -0.2, 0.3, -0.4]
    p2 = [0.5, -0.6, 0.7, -0.8]
    c1, c2 = ga.BlendCrossover(alpha=0.5, seed=42).crossover_real(p1, p2)
    assert len(c1) == len(p1) and len(c2) == len(p2)

    perm1 = [0, 1, 2, 3, 4, 5]
    perm2 = [5, 4, 3, 2, 1, 0]
    o1, o2 = ga.OrderCrossover(seed=42).crossover_perm(perm1, perm2)
    assert sorted(o1) == sorted(perm1) and sorted(o2) == sorted(perm2)

    bits = [True, False, True, False]
    bits2 = ga.BitFlipMutation(seed=42).mutate_bits(bits, pm=0.5)
    assert len(bits2) == len(bits)

    ints = [0, 1, 2, 3, 4]
    ints2 = ga.RandomResettingMutation(seed=42).mutate_int(ints, pm=0.5, min_val=-2, max_val=2)
    assert len(ints2) == len(ints)

    # Registry: create operators by string key
    xreg = ga.CrossoverRegistry()
    xreg.register_factory("blend", lambda: ga.BlendCrossover(alpha=0.25, seed=7))
    mreg = ga.MutationRegistry()
    mreg.register_factory("gauss", lambda: ga.make_gaussian_mutation(seed=7))

    cfg2 = ga.Config()
    cfg2.population_size = 10
    cfg2.generations = 5
    cfg2.dimension = 4
    cfg2.bounds = ga.Bounds(-1.0, 1.0)
    cfg2.seed = 7
    engine2 = ga.GeneticAlgorithm(cfg2)
    engine2.set_crossover_operator(xreg.create("blend"))
    engine2.set_mutation_operator(mreg.create("gauss"))
    r2 = engine2.run(sphere_fitness)
    assert len(r2.best_genes) == 4

    # Benchmark suite
    bcfg = ga.BenchmarkConfig()
    bcfg.warmup_iterations = 0
    bcfg.benchmark_iterations = 1
    bcfg.verbose = False
    bench = ga.GABenchmark(bcfg)
    # Keep sanity fast: exercise object + serialization surface without running
    # the full benchmark loops.
    op_results = bench.operator_results()
    assert isinstance(op_results, list)
    csv_path = os.path.join(out_dir, "python_sanity_benchmark.csv")
    bench.export_to_csv(csv_path)
    assert os.path.exists(csv_path)

    # Hybrid + coevolution
    cfg = ga.Config()
    cfg.dimension = 3
    cfg.population_size = 10
    cfg.generations = 4
    hres = ga.HybridOptimizer(cfg).run(sphere_fitness, lambda x: [0.9 * v for v in x], 2)
    assert len(hres.best_genes) == 3

    ccfg = ga.CoevolutionConfig()
    ccfg.generations = 1
    pops = [[ga.Individual() for _ in range(2)], [ga.Individual() for _ in range(2)]]
    cres = ga.CoevolutionEngine(ccfg).run(pops, lambda _: None, lambda _: None)
    assert len(cres) == 2

    # Checkpoint binary, tracker, visualization
    st = ga.CheckpointState()
    st.config = cfg
    st.result = res
    st.generation = 1
    st.rng_state = "smoke"

    bin_path = os.path.join(out_dir, "python_sanity_checkpoint.bin")
    ga.checkpoint_save_binary(bin_path, st)
    loaded = ga.checkpoint_load_binary(bin_path)
    assert loaded.generation == 1

    tracker = ga.ExperimentTracker("sanity")
    tracker.write_config(cfg, os.path.join(out_dir, "python_sanity_config.txt"))
    tracker.write_history_csv(res, os.path.join(out_dir, "python_sanity_history.csv"))
    ga.export_fitness_curve_csv(res.best_history, res.avg_history, os.path.join(out_dir, "python_sanity_curve.csv"))

    print("python bindings sanity: OK")


if __name__ == "__main__":
    main()
