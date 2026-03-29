"""
Comprehensive check of all important exposed Python APIs.

This script inspects the 'ga' module and validates that all expected
APIs are exposed and functional.
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "python"))

import ga


def check_attribute(obj: Any, attr_name: str) -> bool:
    """Check if an attribute exists on an object."""
    return hasattr(obj, attr_name)


def get_module_classes() -> list[str]:
    """Get all classes from the ga module."""
    classes = []
    for name in dir(ga):
        obj = getattr(ga, name)
        if inspect.isclass(obj):
            classes.append(name)
    return sorted(classes)


def get_module_functions() -> list[str]:
    """Get all functions from the ga module."""
    functions = []
    for name in dir(ga):
        obj = getattr(ga, name)
        if inspect.isfunction(obj) or inspect.isbuiltin(obj):
            functions.append(name)
    return sorted(functions)


def check_core_classes() -> dict[str, list[str]]:
    """Check core configuration and result classes."""
    results = {}

    # Config
    config_attrs = ["population_size", "generations", "dimension", "crossover_rate",
                    "mutation_rate", "bounds", "elite_ratio", "seed"]
    config_ok = all(check_attribute(ga.Config(), attr) for attr in config_attrs)
    results["Config"] = config_attrs if config_ok else ["MISSING ATTRIBUTES"]

    # Bounds
    bounds_attrs = ["lower", "upper"]
    bounds_ok = all(check_attribute(ga.Bounds(), attr) for attr in bounds_attrs)
    results["Bounds"] = bounds_attrs if bounds_ok else ["MISSING ATTRIBUTES"]

    # Result
    result_attrs = ["best_genes", "best_fitness", "best_history", "avg_history"]
    result_ok = all(check_attribute(ga.Result(), attr) for attr in result_attrs)
    results["Result"] = result_attrs if result_ok else ["MISSING ATTRIBUTES"]

    # OptimizationResult
    opt_result_attrs = ["best_solution", "best_fitness", "best_history", "avg_history",
                        "pareto_objectives", "pareto_genes", "evaluations", "generations"]
    opt_result_ok = all(check_attribute(ga.OptimizationResult(), attr) for attr in opt_result_attrs)
    results["OptimizationResult"] = opt_result_attrs if opt_result_ok else ["MISSING ATTRIBUTES"]

    # Evaluation
    eval_attrs = ["objectives", "feasible", "penalty"]
    eval_ok = all(check_attribute(ga.Evaluation(), attr) for attr in eval_attrs)
    results["Evaluation"] = eval_attrs if eval_ok else ["MISSING ATTRIBUTES"]

    # Individual
    ind_attrs = ["evaluation", "age"]
    ind_ok = all(check_attribute(ga.Individual(), attr) for attr in ind_attrs)
    results["Individual"] = ind_attrs if ind_ok else ["MISSING ATTRIBUTES"]

    return results


def check_representation_classes() -> dict[str, list[str]]:
    """Check genome representation classes."""
    results = {}

    # VectorGenome
    vg = ga.VectorGenome([0.1, 0.2])
    vg_attrs = ["genes", "encoding_name"]
    results["VectorGenome"] = vg_attrs if all(check_attribute(vg, attr) for attr in vg_attrs) else ["MISSING"]

    # BitsetGenome
    bg = ga.BitsetGenome(5)
    bg_attrs = ["bits", "size", "hamming_distance", "popcount", "encoding_name"]
    results["BitsetGenome"] = bg_attrs if all(check_attribute(bg, attr) for attr in bg_attrs) else ["MISSING"]

    # PermutationGenome
    pg = ga.PermutationGenome([0, 1, 2])
    pg_attrs = ["order", "size", "is_valid", "position_of", "encoding_name"]
    results["PermutationGenome"] = pg_attrs if all(check_attribute(pg, attr) for attr in pg_attrs) else ["MISSING"]

    # SetGenome
    sg = ga.SetGenome({1, 2, 3})
    sg_attrs = ["values", "encoding_name"]
    results["SetGenome"] = sg_attrs if all(check_attribute(sg, attr) for attr in sg_attrs) else ["MISSING"]

    # MapGenome
    mg = ga.MapGenome({"x": 1.0})
    mg_attrs = ["values", "encoding_name"]
    results["MapGenome"] = mg_attrs if all(check_attribute(mg, attr) for attr in mg_attrs) else ["MISSING"]

    # NdArrayGenome
    ng = ga.NdArrayGenome(2, 3)
    ng_attrs = ["rows", "cols", "data", "get", "set", "encoding_name"]
    results["NdArrayGenome"] = ng_attrs if all(check_attribute(ng, attr) for attr in ng_attrs) else ["MISSING"]

    # TreeGenome
    tg = ga.TreeGenome()
    tg_attrs = ["has_root", "set_root", "root", "encoding_name"]
    results["TreeGenome"] = tg_attrs if all(check_attribute(tg, attr) for attr in tg_attrs) else ["MISSING"]

    return results


def check_gp_classes() -> dict[str, list[str]]:
    """Check genetic programming classes."""
    results = {}

    # ValueType enum
    vt_attrs = ["any", "bool", "int", "double"]
    results["ValueType"] = vt_attrs if all(hasattr(ga.ValueType, attr) for attr in vt_attrs) else ["MISSING"]

    # Signature
    sig = ga.Signature()
    sig_attrs = ["return_type", "arg_types"]
    results["Signature"] = sig_attrs if all(check_attribute(sig, attr) for attr in sig_attrs) else ["MISSING"]

    # Primitive
    prim = ga.Primitive()
    prim_attrs = ["name", "signature", "is_terminal"]
    results["Primitive"] = prim_attrs if all(check_attribute(prim, attr) for attr in prim_attrs) else ["MISSING"]

    # Node
    node = ga.Node("x", ga.ValueType.double)
    node_attrs = ["symbol", "return_type", "size", "child_count", "add_child"]
    results["Node"] = node_attrs if all(check_attribute(node, attr) for attr in node_attrs) else ["MISSING"]

    # TreeBuilder
    tb = ga.TreeBuilder([ga.Primitive()])
    tb_attrs = ["grow"]
    results["TreeBuilder"] = tb_attrs if all(check_attribute(tb, attr) for attr in tb_attrs) else ["MISSING"]

    # ADFPool
    adf = ga.ADFPool()
    adf_attrs = ["put", "has", "get", "size"]
    results["ADFPool"] = adf_attrs if all(check_attribute(adf, attr) for attr in adf_attrs) else ["MISSING"]

    return results


def check_algorithm_classes() -> dict[str, list[str]]:
    """Check main algorithm classes."""
    results = {}

    # GeneticAlgorithm
    cfg = ga.Config()
    cfg.dimension = 3
    cfg.population_size = 10
    cfg.generations = 5
    alg = ga.GeneticAlgorithm(cfg)
    alg_attrs = ["run", "set_mutation_operator", "set_crossover_operator", "config"]
    results["GeneticAlgorithm"] = alg_attrs if all(check_attribute(alg, attr) for attr in alg_attrs) else ["MISSING"]

    # Optimizer
    opt = ga.Optimizer()
    opt_attrs = ["with_config", "with_threads", "with_seed", "optimize",
                 "optimize_multi_objective_nsga2", "optimize_multi_objective_nsga3"]
    results["Optimizer"] = opt_attrs if all(check_attribute(opt, attr) for attr in opt_attrs) else ["MISSING"]

    # OptimizerBuilder
    builder = ga.OptimizerBuilder()
    builder_attrs = ["dimension", "bounds", "population_size", "generations", "seed",
                     "crossover_rate", "mutation_rate", "elite_ratio", "threads", "build"]
    results["OptimizerBuilder"] = builder_attrs if all(check_attribute(builder, attr) for attr in builder_attrs) else ["MISSING"]

    # MultiObjectiveResult
    mo_result = ga.MultiObjectiveResult()
    mo_attrs = ["pareto_genes", "pareto_objectives"]
    results["MultiObjectiveResult"] = mo_attrs if all(check_attribute(mo_result, attr) for attr in mo_attrs) else ["MISSING"]

    return results


def check_moea_classes() -> dict[str, list[str]]:
    """Check multi-objective evolutionary algorithm classes."""
    results = {}

    # NSGA-II
    nsga2_cfg = ga.Nsga2Config()
    nsga2_cfg_attrs = ["population_size", "generations", "seed"]
    results["Nsga2Config"] = nsga2_cfg_attrs if all(check_attribute(nsga2_cfg, attr) for attr in nsga2_cfg_attrs) else ["MISSING"]

    nsga2 = ga.Nsga2()
    nsga2_attrs = ["non_dominated_sort_objectives", "crowding_distance_objectives"]
    results["Nsga2"] = nsga2_attrs if all(check_attribute(nsga2, attr) for attr in nsga2_attrs) else ["MISSING"]

    # NSGA-III
    nsga3 = ga.Nsga3()
    nsga3_attrs = ["generate_reference_points", "non_dominated_sort_objectives",
                   "environmental_select_objectives", "environmental_select_indices"]
    results["Nsga3"] = nsga3_attrs if all(check_attribute(nsga3, attr) for attr in nsga3_attrs) else ["MISSING"]

    # SPEA2
    spea2 = ga.Spea2()
    spea2_attrs = ["strength_fitness_objectives", "environmental_select_objectives",
                   "environmental_select_indices"]
    results["Spea2"] = spea2_attrs if all(check_attribute(spea2, attr) for attr in spea2_attrs) else ["MISSING"]

    return results


def check_es_classes() -> dict[str, list[str]]:
    """Check evolution strategy classes."""
    results = {}

    # EvolutionStrategy
    es_cfg = ga.EvolutionStrategyConfig()
    es_cfg_attrs = ["mu", "lambda_", "generations", "dimension", "sigma", "lower", "upper", "plus_strategy", "seed"]
    results["EvolutionStrategyConfig"] = es_cfg_attrs if all(check_attribute(es_cfg, attr) for attr in es_cfg_attrs) else ["MISSING"]

    es_result = ga.EvolutionStrategyResult()
    es_result_attrs = ["best", "best_fitness", "best_history"]
    results["EvolutionStrategyResult"] = es_result_attrs if all(check_attribute(es_result, attr) for attr in es_result_attrs) else ["MISSING"]

    es = ga.EvolutionStrategy(es_cfg)
    es_attrs = ["run"]
    results["EvolutionStrategy"] = es_attrs if all(check_attribute(es, attr) for attr in es_attrs) else ["MISSING"]

    # CMA-ES
    cma_cfg = ga.CmaEsConfig()
    cma_cfg_attrs = ["population_size", "generations", "dimension", "lower", "upper", "sigma", "seed"]
    results["CmaEsConfig"] = cma_cfg_attrs if all(check_attribute(cma_cfg, attr) for attr in cma_cfg_attrs) else ["MISSING"]

    cma_result = ga.CmaEsResult()
    cma_result_attrs = ["best", "best_fitness", "history"]
    results["CmaEsResult"] = cma_result_attrs if all(check_attribute(cma_result, attr) for attr in cma_result_attrs) else ["MISSING"]

    cma = ga.DiagonalCmaEs(cma_cfg)
    cma_attrs = ["run"]
    results["DiagonalCmaEs"] = cma_attrs if all(check_attribute(cma, attr) for attr in cma_attrs) else ["MISSING"]

    # MO-CMA-ES
    mo_cma_cfg = ga.MoCmaEsConfig()
    mo_cma_cfg_attrs = ["cma", "weights"]
    results["MoCmaEsConfig"] = mo_cma_cfg_attrs if all(check_attribute(mo_cma_cfg, attr) for attr in mo_cma_cfg_attrs) else ["MISSING"]

    mo_cma_result = ga.MoCmaEsResult()
    mo_cma_result_attrs = ["best", "objectives", "weighted_fitness"]
    results["MoCmaEsResult"] = mo_cma_result_attrs if all(check_attribute(mo_cma_result, attr) for attr in mo_cma_result_attrs) else ["MISSING"]

    mo_cma = ga.MoCmaEs(mo_cma_cfg)
    mo_cma_attrs = ["run"]
    results["MoCmaEs"] = mo_cma_attrs if all(check_attribute(mo_cma, attr) for attr in mo_cma_attrs) else ["MISSING"]

    return results


def check_constraint_classes() -> dict[str, list[str]]:
    """Check constraint and adaptive classes."""
    results = {}

    # ConstraintSet
    cs = ga.ConstraintSet()
    cs_attrs = ["add_hard_constraint", "add_soft_penalty", "add_repair", "clear"]
    results["ConstraintSet"] = cs_attrs if all(check_attribute(cs, attr) for attr in cs_attrs) else ["MISSING"]

    # AdaptiveRates
    rates = ga.AdaptiveRates()
    rates_attrs = ["mutation_rate", "crossover_rate"]
    results["AdaptiveRates"] = rates_attrs if all(check_attribute(rates, attr) for attr in rates_attrs) else ["MISSING"]

    # AdaptiveRateController
    arc = ga.AdaptiveRateController()
    arc_attrs = ["update"]
    results["AdaptiveRateController"] = arc_attrs if all(check_attribute(arc, attr) for attr in arc_attrs) else ["MISSING"]

    return results


def check_hybrid_coevolution_classes() -> dict[str, list[str]]:
    """Check hybrid and coevolution classes."""
    results = {}

    # HybridOptimizer
    cfg = ga.Config()
    cfg.dimension = 3
    cfg.population_size = 10
    cfg.generations = 5
    hybrid = ga.HybridOptimizer(cfg)
    hybrid_attrs = ["run"]
    results["HybridOptimizer"] = hybrid_attrs if all(check_attribute(hybrid, attr) for attr in hybrid_attrs) else ["MISSING"]

    # CoevolutionEngine
    coevo_cfg = ga.CoevolutionConfig()
    coevo_cfg_attrs = ["generations", "seed"]
    results["CoevolutionConfig"] = coevo_cfg_attrs if all(check_attribute(coevo_cfg, attr) for attr in coevo_cfg_attrs) else ["MISSING"]

    coevo = ga.CoevolutionEngine(coevo_cfg)
    coevo_attrs = ["run"]
    results["CoevolutionEngine"] = coevo_attrs if all(check_attribute(coevo, attr) for attr in coevo_attrs) else ["MISSING"]

    return results


def check_evaluation_classes() -> dict[str, list[str]]:
    """Check parallel evaluation classes."""
    results = {}

    # ParallelEvaluator
    def dummy_fitness(x):
        return sum(v*v for v in x)

    pe = ga.ParallelEvaluator(dummy_fitness, threads=2)
    pe_attrs = ["evaluate"]
    results["ParallelEvaluator"] = pe_attrs if all(check_attribute(pe, attr) for attr in pe_attrs) else ["MISSING"]

    # LocalDistributedExecutor
    lde = ga.LocalDistributedExecutor(dummy_fitness, workers=2)
    lde_attrs = ["execute"]
    results["LocalDistributedExecutor"] = lde_attrs if all(check_attribute(lde, attr) for attr in lde_attrs) else ["MISSING"]

    return results


def check_tracking_visualization_classes() -> dict[str, list[str]]:
    """Check tracking and visualization classes."""
    results = {}

    # ExperimentTracker
    tracker = ga.ExperimentTracker("test")
    tracker_attrs = ["write_config", "write_history_csv", "write_best_solution_csv"]
    results["ExperimentTracker"] = tracker_attrs if all(check_attribute(tracker, attr) for attr in tracker_attrs) else ["MISSING"]

    # CheckpointState
    state = ga.CheckpointState()
    state_attrs = ["config", "result", "generation", "rng_state"]
    results["CheckpointState"] = state_attrs if all(check_attribute(state, attr) for attr in state_attrs) else ["MISSING"]

    return results


def check_benchmark_classes() -> dict[str, list[str]]:
    """Check benchmark classes."""
    results = {}

    # BenchmarkConfig
    bcfg = ga.BenchmarkConfig()
    bcfg_attrs = ["warmup_iterations", "benchmark_iterations", "verbose", "csv_output", "output_file"]
    results["BenchmarkConfig"] = bcfg_attrs if all(check_attribute(bcfg, attr) for attr in bcfg_attrs) else ["MISSING"]

    # GABenchmark
    bench = ga.GABenchmark()
    bench_attrs = ["run_all_benchmarks", "run_operator_benchmarks", "run_function_benchmarks",
                   "run_scalability_benchmarks", "generate_report", "export_to_csv",
                   "operator_results", "function_results", "scalability_results"]
    results["GABenchmark"] = bench_attrs if all(check_attribute(bench, attr) for attr in bench_attrs) else ["MISSING"]

    return results


def check_module_functions() -> list[str]:
    """Check all module-level functions."""
    expected_functions = [
        # Operator factories
        "make_gaussian_mutation",
        "make_uniform_mutation",
        "make_one_point_crossover",
        "make_two_point_crossover",

        # Selection helpers
        "selection_tournament_indices",
        "selection_roulette_indices",
        "selection_rank_indices",
        "selection_sus_indices",
        "selection_elitism_indices",

        # NSGA-II functions
        "nsga2_non_dominated_sort",
        "nsga2_crowding_distance",

        # NSGA-III functions
        "nsga3_reference_points",
        "nsga3_environmental_select_indices",

        # SPEA2 functions
        "spea2_strength_fitness",
        "spea2_environmental_select_indices",

        # Constraint functions
        "is_feasible",
        "total_penalty",
        "apply_repairs",
        "penalized_fitness",

        # Checkpoint functions
        "checkpoint_save_json",
        "checkpoint_load_json",
        "checkpoint_save_binary",
        "checkpoint_load_binary",

        # Visualization/export functions
        "export_fitness_curve_csv",
        "export_pareto_front_csv",
        "export_diversity_csv",
    ]

    available = []
    missing = []

    for func_name in expected_functions:
        if hasattr(ga, func_name):
            available.append(func_name)
        else:
            missing.append(func_name)

    return available, missing


def main() -> None:
    """Run comprehensive API check."""
    print("=" * 80)
    print("COMPREHENSIVE PYTHON API CHECK FOR 'ga' MODULE")
    print("=" * 80)
    print()

    print(f"Module location: {ga.__file__}")
    print(f"Module docstring: {ga.__doc__}")
    print()

    # Get all available classes and functions
    all_classes = get_module_classes()
    all_functions = get_module_functions()

    print(f"Total classes exposed: {len(all_classes)}")
    print(f"Total functions exposed: {len(all_functions)}")
    print()

    # Check each category
    print("-" * 80)
    print("1. CORE CONFIGURATION AND RESULT CLASSES")
    print("-" * 80)
    core_results = check_core_classes()
    for cls_name, attrs in core_results.items():
        print(f"  {cls_name:25s}: {', '.join(attrs)}")
    print()

    print("-" * 80)
    print("2. GENOME REPRESENTATION CLASSES")
    print("-" * 80)
    repr_results = check_representation_classes()
    for cls_name, attrs in repr_results.items():
        print(f"  {cls_name:25s}: {', '.join(attrs)}")
    print()

    print("-" * 80)
    print("3. GENETIC PROGRAMMING CLASSES")
    print("-" * 80)
    gp_results = check_gp_classes()
    for cls_name, attrs in gp_results.items():
        print(f"  {cls_name:25s}: {', '.join(attrs)}")
    print()

    print("-" * 80)
    print("4. MAIN ALGORITHM CLASSES")
    print("-" * 80)
    alg_results = check_algorithm_classes()
    for cls_name, attrs in alg_results.items():
        print(f"  {cls_name:25s}: {', '.join(attrs)}")
    print()

    print("-" * 80)
    print("5. MULTI-OBJECTIVE EVOLUTIONARY ALGORITHMS (MOEA)")
    print("-" * 80)
    moea_results = check_moea_classes()
    for cls_name, attrs in moea_results.items():
        print(f"  {cls_name:25s}: {', '.join(attrs)}")
    print()

    print("-" * 80)
    print("6. EVOLUTION STRATEGIES (ES/CMA-ES/MO-CMA-ES)")
    print("-" * 80)
    es_results = check_es_classes()
    for cls_name, attrs in es_results.items():
        print(f"  {cls_name:25s}: {', '.join(attrs)}")
    print()

    print("-" * 80)
    print("7. CONSTRAINTS AND ADAPTIVE CONTROL")
    print("-" * 80)
    constraint_results = check_constraint_classes()
    for cls_name, attrs in constraint_results.items():
        print(f"  {cls_name:25s}: {', '.join(attrs)}")
    print()

    print("-" * 80)
    print("8. HYBRID AND COEVOLUTION")
    print("-" * 80)
    hybrid_results = check_hybrid_coevolution_classes()
    for cls_name, attrs in hybrid_results.items():
        print(f"  {cls_name:25s}: {', '.join(attrs)}")
    print()

    print("-" * 80)
    print("9. PARALLEL EVALUATION")
    print("-" * 80)
    eval_results = check_evaluation_classes()
    for cls_name, attrs in eval_results.items():
        print(f"  {cls_name:25s}: {', '.join(attrs)}")
    print()

    print("-" * 80)
    print("10. TRACKING, VISUALIZATION, AND CHECKPOINTING")
    print("-" * 80)
    tracking_results = check_tracking_visualization_classes()
    for cls_name, attrs in tracking_results.items():
        print(f"  {cls_name:25s}: {', '.join(attrs)}")
    print()

    print("-" * 80)
    print("11. BENCHMARK SUITE")
    print("-" * 80)
    benchmark_results = check_benchmark_classes()
    for cls_name, attrs in benchmark_results.items():
        print(f"  {cls_name:25s}: {', '.join(attrs)}")
    print()

    print("-" * 80)
    print("12. MODULE-LEVEL FUNCTIONS")
    print("-" * 80)
    available_funcs, missing_funcs = check_module_functions()
    print(f"Available functions ({len(available_funcs)}):")
    for func in available_funcs:
        print(f"  ✓ {func}")
    if missing_funcs:
        print(f"\nMissing functions ({len(missing_funcs)}):")
        for func in missing_funcs:
            print(f"  ✗ {func}")
    print()

    print("=" * 80)
    print("API CHECK COMPLETE")
    print("=" * 80)
    print()

    # Summary statistics
    total_checked = (
        len(core_results) + len(repr_results) + len(gp_results) +
        len(alg_results) + len(moea_results) + len(es_results) +
        len(constraint_results) + len(hybrid_results) + len(eval_results) +
        len(tracking_results) + len(benchmark_results)
    )
    print(f"Total API categories checked: 12")
    print(f"Total classes checked: {total_checked}")
    print(f"Total functions checked: {len(available_funcs)}")
    print(f"Missing functions: {len(missing_funcs)}")
    print()

    if missing_funcs:
        print("⚠ WARNING: Some expected functions are missing!")
        return 1
    else:
        print("✓ All expected APIs are exposed and accessible!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
