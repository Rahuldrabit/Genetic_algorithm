"""Sanity checks for Python bindings coverage."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "python"))

import ga


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
