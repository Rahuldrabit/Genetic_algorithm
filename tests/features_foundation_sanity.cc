#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "ga/adaptive/adaptive_policy.hpp"
#include "ga/api/optimizer.hpp"
#include "ga/checkpoint/checkpoint.hpp"
#include "ga/coevolution/coevolution.hpp"
#include "ga/constraints/constraints.hpp"
#include "ga/es/cmaes.hpp"
#include "ga/es/evolution_strategies.hpp"
#include "ga/evaluation/distributed_executor.hpp"
#include "ga/evaluation/parallel_evaluator.hpp"
#include "ga/gp/adf.hpp"
#include "ga/gp/tree_builder.hpp"
#include "ga/hybrid/hybrid_optimizer.hpp"
#include "ga/moea/mo_cmaes.hpp"
#include "ga/moea/nsga3.hpp"
#include "ga/moea/spea2.hpp"
#include "ga/plugin/registry.hpp"
#include "ga/representations/map_genome.hpp"
#include "ga/representations/ndarray_genome.hpp"
#include "ga/representations/set_genome.hpp"
#include "ga/representations/tree_genome.hpp"
#include "ga/representations/vector_genome.hpp"
#include "ga/tracking/experiment_tracker.hpp"
#include "ga/visualization/export.hpp"

namespace fs = std::filesystem;

namespace {

bool approx(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) <= eps;
}

} // namespace

int main() {
    try {
        // Representations
        ga::representations::VectorGenome<double> vg({1.0, 2.0, 3.0});
        ga::representations::SetGenome sg({1, 2, 3});
        ga::representations::MapGenome mg({{"x", 1.0}, {"y", 2.0}});
        ga::representations::NdArrayGenome ng(2, 2);
        ng.at(0, 1) = 42.0;
        if (!approx(ng.at(0, 1), 42.0)) {
            throw std::runtime_error("ndarray genome sanity failed");
        }

        // GP typed + loose build and ADF
        std::vector<ga::gp::Primitive> prim = {
            {"+", {ga::gp::ValueType::Double, {ga::gp::ValueType::Double, ga::gp::ValueType::Double}}, false},
            {"x", {ga::gp::ValueType::Double, {}}, true},
            {"1.0", {ga::gp::ValueType::Double, {}}, true}
        };
        ga::gp::TreeBuilder builder(prim);
        std::mt19937 gp_rng(7);
        auto strongTree = builder.grow(3, ga::gp::ValueType::Double, true, gp_rng);
        auto looseTree = builder.grow(3, ga::gp::ValueType::Any, false, gp_rng);
        if (!strongTree || !looseTree) {
            throw std::runtime_error("GP tree build failed");
        }
        ga::gp::ADFPool adf;
        adf.put("f0", *strongTree);
        if (!adf.has("f0")) {
            throw std::runtime_error("ADF pool failed");
        }
        ga::representations::TreeGenome tg(strongTree->clone());
        (void)tg;

        // Parallel evaluator
        ga::evaluation::ParallelEvaluator<int, int, std::function<int(const int&)>> pe(
            [](const int& x) { return x * x; }, 2);
        auto pe_out = pe.evaluate({1, 2, 3, 4});
        if (pe_out.size() != 4 || pe_out[3] != 16) {
            throw std::runtime_error("Parallel evaluator failed");
        }

        ga::evaluation::LocalDistributedExecutor dist_eval(
            [](const std::vector<double>& x) {
                double s = 0.0;
                for (double v : x) {
                    s += v * v;
                }
                return s;
            },
            2);
        auto dist_out = dist_eval.execute({{1.0, 2.0}, {3.0, 4.0}});
        if (dist_out.size() != 2 || !approx(dist_out[0], 5.0) || !approx(dist_out[1], 25.0)) {
            throw std::runtime_error("distributed evaluator failed");
        }

#if defined(__unix__) || defined(__APPLE__)
        ga::evaluation::ProcessDistributedExecutor proc_eval(
            [](const std::vector<double>& x) {
                double s = 0.0;
                for (double v : x) {
                    s += v * v;
                }
                return s;
            },
            2);
        auto proc_out = proc_eval.execute({{1.0, 2.0}, {3.0, 4.0}, {0.0, 5.0}});
        if (proc_out.size() != 3 || !approx(proc_out[0], 5.0) || !approx(proc_out[1], 25.0) ||
            !approx(proc_out[2], 25.0)) {
            throw std::runtime_error("process distributed evaluator failed");
        }
#endif

        // ES / CMA-ES
        ga::es::EvolutionStrategy es({10, 30, 12, 5, 0.2, -2.0, 2.0, true, 11});
        auto es_res = es.run([](const std::vector<double>& x) {
            double s = 0.0;
            for (double v : x) s += v * v;
            return 1000.0 / (1.0 + s);
        });
        if (es_res.best.empty()) {
            throw std::runtime_error("ES result empty");
        }

        ga::es::DiagonalCmaEs cma({24, 20, 5, -2.0, 2.0, 0.2, 9});
        auto cma_res = cma.run([](const std::vector<double>& x) {
            double s = 0.0;
            for (double v : x) s += v * v;
            return 1000.0 / (1.0 + s);
        });
        if (cma_res.best.empty()) {
            throw std::runtime_error("CMA-ES result empty");
        }

        // MOEA modules
        ga::moea::Spea2 spea2;
        std::vector<ga::Individual> mopop(4);
        mopop[0].evaluation.objectives = {1.0, 3.0};
        mopop[1].evaluation.objectives = {2.0, 2.0};
        mopop[2].evaluation.objectives = {3.0, 1.0};
        mopop[3].evaluation.objectives = {2.5, 2.5};
        auto sf = spea2.strengthFitness(mopop);
        if (sf.size() != mopop.size()) {
            throw std::runtime_error("SPEA2 fitness size mismatch");
        }

        ga::moea::Nsga3 nsga3;
        auto selected = nsga3.environmentalSelect(mopop, 2, {{0.0, 1.0}, {1.0, 0.0}});
        if (selected.size() != 2) {
            throw std::runtime_error("NSGA-III selection failed");
        }

        ga::moea::MoCmaEs mocma({{20, 8, 4, -2.0, 2.0, 0.2, 3}, {0.5, 0.5}});
        auto mocma_res = mocma.run([](const std::vector<double>& x) {
            double f1 = 0.0;
            for (double v : x) f1 += v * v;
            double f2 = 0.0;
            for (double v : x) f2 += (v - 1.0) * (v - 1.0);
            return std::vector<double>{f1, f2};
        });
        if (mocma_res.best.empty()) {
            throw std::runtime_error("MO-CMA-ES result empty");
        }

        // Co-evolution
        ga::coevolution::CoevolutionEngine coevo({3, 13});
        ga::coevolution::CoevolutionEngine::Populations pops(2);
        pops[0].resize(4);
        pops[1].resize(4);
        auto final_pops = coevo.run(
            std::move(pops),
            [](ga::coevolution::CoevolutionEngine::Populations& populations) {
                for (auto& pop : populations) {
                    for (auto& ind : pop) {
                        ind.evaluation.objectives = {1.0};
                    }
                }
            },
            [](ga::coevolution::CoevolutionEngine::Populations& populations, std::mt19937&) {
                for (auto& pop : populations) {
                    for (auto& ind : pop) {
                        ind.age += 1;
                    }
                }
            });
        if (final_pops.size() != 2) {
            throw std::runtime_error("coevolution failed");
        }

        // Adaptive
        ga::adaptive::AdaptiveRateController adaptive;
        auto rates = adaptive.update({0.1, 0.8}, 0.1, 0.0);
        if (rates.mutationRate < 0.1) {
            throw std::runtime_error("adaptive update unexpected");
        }

        // Constraints
        ga::constraints::ConstraintSet cset;
        cset.hard.push_back([](const std::vector<double>& g) { return g[0] <= 1.0; });
        cset.soft.push_back([](const std::vector<double>& g) { return std::max(0.0, g[1] - 1.0); });
        cset.repairs.push_back([](std::vector<double>& g) { g[0] = std::min(g[0], 1.0); });

        std::vector<double> genes = {2.0, 3.0};
        ga::constraints::applyRepairs(genes, cset);
        if (!ga::constraints::isFeasible(genes, cset)) {
            throw std::runtime_error("constraint repair failed");
        }

        // Hybrid optimizer
        ga::Config cfg;
        cfg.populationSize = 20;
        cfg.generations = 10;
        cfg.dimension = 3;
        cfg.bounds = {-2.0, 2.0};
        cfg.seed = 4;
        ga::hybrid::HybridOptimizer hybrid(cfg);
        auto hres = hybrid.run(
            [](const std::vector<double>& x) {
                double s = 0.0;
                for (double v : x) s += v * v;
                return 1000.0 / (1.0 + s);
            },
            [](std::vector<double>& x) {
                for (double& v : x) v *= 0.8;
            },
            3);
        if (hres.bestGenes.empty()) {
            throw std::runtime_error("hybrid optimizer failed");
        }

        // Plugin registry
        struct Base {
            virtual ~Base() = default;
            virtual int value() const = 0;
        };
        struct Impl : Base {
            int value() const override { return 7; }
        };
        ga::plugin::Registry<Base> reg;
        reg.registerFactory("impl", [] { return std::make_unique<Impl>(); });
        auto obj = reg.create("impl");
        if (obj->value() != 7) {
            throw std::runtime_error("plugin registry failed");
        }

        // Checkpoint + tracker + visualization
        fs::create_directories("build/tests/tmp");

        ga::checkpoint::CheckpointState cp;
        cp.config = cfg;
        cp.generation = 5;
        cp.rngState = "rng\\state\"ok\nline";
        cp.result = hres;
        const std::string cp_path = "build/tests/tmp/checkpoint.bin";
        ga::checkpoint::CheckpointManager::saveBinary(cp_path, cp);
        auto cp_load = ga::checkpoint::CheckpointManager::loadBinary(cp_path);
        if (cp_load.generation != cp.generation) {
            throw std::runtime_error("checkpoint roundtrip failed");
        }

        const std::string cp_json_path = "build/tests/tmp/checkpoint.json";
        ga::checkpoint::CheckpointManager::saveJson(cp_json_path, cp);
        auto cp_json_load = ga::checkpoint::CheckpointManager::loadJson(cp_json_path);
        if (cp_json_load.generation != cp.generation ||
            cp_json_load.rngState != cp.rngState ||
            cp_json_load.result.bestGenes.size() != cp.result.bestGenes.size()) {
            throw std::runtime_error("checkpoint JSON roundtrip failed");
        }

        ga::tracking::ExperimentTracker tracker("run-1");
        tracker.writeConfig(cfg, "build/tests/tmp/config.txt");
        tracker.writeHistoryCSV(hres, "build/tests/tmp/history.csv");
        tracker.writeBestSolutionCSV(hres, "build/tests/tmp/best.csv");

        ga::visualization::exportFitnessCurveCSV(hres.bestHistory, hres.avgHistory, "build/tests/tmp/fit.csv");
        ga::visualization::exportParetoFrontCSV({{1.0, 2.0}, {2.0, 1.0}}, "build/tests/tmp/pareto.csv");
        ga::visualization::exportDiversityCSV({0.2, 0.3, 0.4}, "build/tests/tmp/div.csv");

        // High-level API
        ga::api::Optimizer optimizer;
        optimizer.withConfig(cfg).withSeed(4).withThreads(2);
        auto ores = optimizer.optimize([](const std::vector<double>& x) {
            double s = 0.0;
            for (double v : x) s += v * v;
            return 1000.0 / (1.0 + s);
        });
        if (ores.bestGenes.empty()) {
            throw std::runtime_error("high-level optimize failed");
        }
        auto mores = optimizer.optimizeMultiObjective({
            [](const std::vector<double>& x) {
                double s = 0.0;
                for (double v : x) s += v * v;
                return s;
            },
            [](const std::vector<double>& x) {
                double s = 0.0;
                for (double v : x) s += (v - 1.0) * (v - 1.0);
                return s;
            }
        }, 20, 10);
        if (mores.paretoGenes.empty()) {
            throw std::runtime_error("high-level multi-objective optimize failed");
        }

        auto mores3 = optimizer.optimizeMultiObjectiveNsga3({
            [](const std::vector<double>& x) {
                double s = 0.0;
                for (double v : x) s += v * v;
                return s;
            },
            [](const std::vector<double>& x) {
                double s = 0.0;
                for (double v : x) s += (v - 1.0) * (v - 1.0);
                return s;
            }
        },
                                                            20,
                                                            10,
                                                            6);
        if (mores3.paretoGenes.empty()) {
            throw std::runtime_error("high-level NSGA-III optimize failed");
        }

        std::cout << "[PASS] Feature foundation sanity checks\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}
