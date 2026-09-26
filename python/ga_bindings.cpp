/**
 * Python bindings for the Genetic Algorithm framework using pybind11.
 *
 * Exposes (via `import genetic_algorithm_lib as ga`):
 *  - ga.Config        - Algorithm configuration
 *  - ga.Bounds        - Gene bounds (lower, upper)
 *  - ga.Result        - Run results
 *  - ga.GeneticAlgorithm - Main GA class
 *  - ga.ParticleSwarmOptimizer / AntColonyOptimizer / GSA / ACOR
 *  - ga.FuzzyCMeans / FuzzyAdaptiveController
 *  - ga.MetaheuristicPipeline - User-ordered optimizer composition
 *  - Operator factories: make_gaussian_mutation, make_uniform_mutation,
 *                        make_one_point_crossover, make_two_point_crossover
 */
#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/optional.h>

#include <random>
#include <memory>
#include <stdexcept>
#include <thread>
#include <limits>

#include "ga/config.hpp"
#include "ga/genetic_algorithm.hpp"
#include "ga/api/builder.hpp"
#include "ga/api/optimizer.hpp"
#include "ga/adaptive/adaptive_policy.hpp"
#include "ga/algorithms/moea/nsga2.hpp"
#include "ga/checkpoint/checkpoint.hpp"
#include "ga/constraints/constraints.hpp"
#include "ga/coevolution/coevolution.hpp"
#include "ga/evaluation/distributed_executor.hpp"
#include "ga/evaluation/parallel_evaluator.hpp"
#include "ga/core/evaluation.hpp"
#include "ga/core/genome.hpp"
#include "ga/core/individual.hpp"
#include "ga/core/result.hpp"
#include "ga/es/cmaes.hpp"
#include "ga/es/evolution_strategies.hpp"
#include "ga/gp/adf.hpp"
#include "ga/gp/tree_builder.hpp"
#include "ga/gp/type_system.hpp"
#include "ga/hybrid/hybrid_optimizer.hpp"
#include "ga/metaheuristics.hpp"
#include "ga/moea/nsga3.hpp"
#include "ga/moea/mo_cmaes.hpp"
#include "ga/moea/spea2.hpp"
#include "ga/plugin/registry.hpp"
#include "ga/representations/bitset_genome.hpp"
#include "ga/representations/map_genome.hpp"
#include "ga/representations/ndarray_genome.hpp"
#include "ga/representations/permutation_genome.hpp"
#include "ga/representations/set_genome.hpp"
#include "ga/representations/tree_genome.hpp"
#include "ga/representations/vector_genome.hpp"
#include "ga/tracking/experiment_tracker.hpp"
#include "ga/visualization/export.hpp"

// Full type definitions needed by pybind11 for operator ownership transfer
#include "mutation/base_mutation.h"
#include "mutation/bit_flip_mutation.h"
#include "mutation/creep_mutation.h"
#include "mutation/gaussian_mutation.h"
#include "mutation/insert_mutation.h"
#include "mutation/inversion_mutation.h"
#include "mutation/list_mutation.h"
#include "mutation/random_resetting_mutation.h"
#include "mutation/scramble_mutation.h"
#include "mutation/self_adaptive_mutation.h"
#include "mutation/swap_mutation.h"
#include "mutation/uniform_mutation.h"
#include "crossover/base_crossover.h"
#include "crossover/blend_crossover.h"
#include "crossover/cut_and_crossfill_crossover.h"
#include "crossover/cycle_crossover.h"
#include "crossover/differential_evolution_crossover.h"
#include "crossover/diploid_recombination.h"
#include "crossover/edge_crossover.h"
#include "crossover/intermediate_recombination.h"
#include "crossover/line_recombination.h"
#include "crossover/multi_point_crossover.h"
#include "crossover/one_point_crossover.h"
#include "crossover/order_crossover.h"
#include "crossover/partially_mapped_crossover.h"
#include "crossover/simulated_binary_crossover.h"
#include "crossover/two_point_crossover.h"
#include "crossover/uniform_crossover.h"
#include "crossover/uniform_k_vector_crossover.h"
#include "selection-operator/tournament_selection.h"
#include "selection-operator/roulette_wheel_selection.h"
#include "selection-operator/rank_selection.h"
#include "selection-operator/stochastic_universal_sampling.h"
#include "selection-operator/elitism_selection.h"
#include "benchmark/ga_benchmark.h"

namespace nb = nanobind;
using namespace nb::literals;

static std::vector<ga::Individual> objectivesToIndividuals(
    const std::vector<std::vector<double>>& objectiveMatrix,
    bool tagIndex = false) {
    std::vector<ga::Individual> population;
    population.reserve(objectiveMatrix.size());

    for (std::size_t i = 0; i < objectiveMatrix.size(); ++i) {
        const auto& objectives = objectiveMatrix[i];
        if (objectives.empty()) {
            throw std::invalid_argument("Objective vectors must be non-empty");
        }
        ga::Individual ind;
        ind.evaluation.objectives = objectives;
        if (tagIndex) {
            ind.age = static_cast<int>(i);
        }
        population.push_back(std::move(ind));
    }
    return population;
}

static std::vector<std::vector<double>> individualsToObjectives(
    const std::vector<ga::Individual>& individuals) {
    std::vector<std::vector<double>> out;
    out.reserve(individuals.size());
    for (const auto& ind : individuals) {
        out.push_back(ind.evaluation.objectives);
    }
    return out;
}

static std::vector<ga::api::Optimizer::Objective> pyObjectivesToCpp(const nb::iterable& objectiveCallables) {
    std::vector<ga::api::Optimizer::Objective> objectives;
    for (nb::handle h : objectiveCallables) {
        nb::object obj = nb::borrow<nb::object>(h);
        objectives.emplace_back([obj](const std::vector<double>& genes) {
            nb::gil_scoped_acquire acquire;
            return nb::cast<double>(obj(genes));
        });
    }
    return objectives;
}

static std::vector<::Individual> fitnessToSelectionPopulation(const std::vector<double>& fitness) {
    std::vector<::Individual> population;
    population.reserve(fitness.size());
    for (double f : fitness) {
        population.emplace_back(f);
    }
    return population;
}

static unsigned int checkedCountToUInt(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
        throw std::out_of_range(std::string(name) + " exceeds unsigned int range");
    }
    return static_cast<unsigned int>(value);
}

using DoubleBatchEvaluator =
    ga::evaluation::ParallelEvaluator<std::vector<double>,
                                      double,
                                      std::function<double(const std::vector<double>&)>>;

class PyAdaptiveController : public ga::metaheuristics::IAdaptiveController {
public:
    NB_TRAMPOLINE(ga::metaheuristics::IAdaptiveController);

    ga::metaheuristics::ControlSignal update(
        const ga::metaheuristics::ProgressState& state) const override {
        NB_OVERRIDE_PURE(update, state);
    }
};

class PyContinuousOptimizer : public ga::metaheuristics::IContinuousOptimizer {
public:
    NB_TRAMPOLINE(ga::metaheuristics::IContinuousOptimizer);

    std::string name() const override {
        NB_OVERRIDE_PURE(name);
    }

    ga::core::OptimizationResult optimize(
        const ga::Fitness& fitness,
        const ga::metaheuristics::SeedPopulation& seeds) override {
        NB_OVERRIDE_PURE(optimize, fitness, seeds);
    }
};

static ga::Fitness wrapPythonFitness(nb::callable fitness) {
    return [fitness = std::move(fitness)](const std::vector<double>& genes) {
        nb::gil_scoped_acquire acquire;
        return nb::cast<double>(fitness(genes));
    };
}

template <typename Optimizer>
static ga::core::OptimizationResult optimizeFromPython(
    Optimizer& optimizer,
    nb::callable fitness,
    const ga::metaheuristics::SeedPopulation& seeds) {
    ga::Fitness wrapped = wrapPythonFitness(std::move(fitness));
    nb::gil_scoped_release release;
    return optimizer.optimize(wrapped, seeds);
}

static std::shared_ptr<ga::metaheuristics::IAdaptiveController> mutableController(
    const std::shared_ptr<const ga::metaheuristics::IAdaptiveController>& controller) {
    return std::const_pointer_cast<ga::metaheuristics::IAdaptiveController>(controller);
}

class PyCrossoverOperatorWrapper : public CrossoverOperator {
    nb::object obj_;
    CrossoverOperator* raw_op_{nullptr};
public:
    PyCrossoverOperatorWrapper(nb::object obj, CrossoverOperator* op)
        : CrossoverOperator(op ? op->getName() : "PyCrossover"),
          obj_(std::move(obj)),
          raw_op_(op) {}

    ~PyCrossoverOperatorWrapper() override {
        nb::gil_scoped_acquire acquire;
        obj_.reset();
    }

    std::pair<RealVector, RealVector> crossover(const RealVector& p1, const RealVector& p2) override {
        if (raw_op_) {
            return raw_op_->crossover(p1, p2);
        }
        nb::gil_scoped_acquire acquire;
        if (nb::hasattr(obj_, "crossover_real")) {
            return nb::cast<std::pair<RealVector, RealVector>>(obj_.attr("crossover_real")(p1, p2));
        }
        return CrossoverOperator::crossover(p1, p2);
    }

    std::pair<BitString, BitString> crossover(const BitString& p1, const BitString& p2) override {
        if (raw_op_) {
            return raw_op_->crossover(p1, p2);
        }
        nb::gil_scoped_acquire acquire;
        if (nb::hasattr(obj_, "crossover_bits")) {
            return nb::cast<std::pair<BitString, BitString>>(obj_.attr("crossover_bits")(p1, p2));
        }
        return CrossoverOperator::crossover(p1, p2);
    }

    std::pair<IntVector, IntVector> crossover(const IntVector& p1, const IntVector& p2) override {
        if (raw_op_) {
            return raw_op_->crossover(p1, p2);
        }
        nb::gil_scoped_acquire acquire;
        if (nb::hasattr(obj_, "crossover_int")) {
            return nb::cast<std::pair<IntVector, IntVector>>(obj_.attr("crossover_int")(p1, p2));
        }
        return CrossoverOperator::crossover(p1, p2);
    }
};

class PyMutationOperatorWrapper : public MutationOperator {
    nb::object obj_;
public:
    explicit PyMutationOperatorWrapper(nb::object obj)
        : MutationOperator("PyMutation"), obj_(std::move(obj)) {}
    ~PyMutationOperatorWrapper() override {
        nb::gil_scoped_acquire acquire;
        obj_.reset();
    }
};

static inline std::unique_ptr<CrossoverOperator> cloneCrossover(nb::handle h) {
    if (!h.is_valid() || h.is_none()) return nullptr;
    if (nb::isinstance<BlendCrossover>(h)) return std::make_unique<BlendCrossover>(nb::cast<const BlendCrossover&>(h));
    if (nb::isinstance<OnePointCrossover>(h)) return std::make_unique<OnePointCrossover>(nb::cast<const OnePointCrossover&>(h));
    if (nb::isinstance<TwoPointCrossover>(h)) return std::make_unique<TwoPointCrossover>(nb::cast<const TwoPointCrossover&>(h));
    if (nb::isinstance<UniformCrossover>(h)) return std::make_unique<UniformCrossover>(nb::cast<const UniformCrossover&>(h));
    if (nb::isinstance<MultiPointCrossover>(h)) return std::make_unique<MultiPointCrossover>(nb::cast<const MultiPointCrossover&>(h));
    if (nb::isinstance<SimulatedBinaryCrossover>(h)) return std::make_unique<SimulatedBinaryCrossover>(nb::cast<const SimulatedBinaryCrossover&>(h));
    if (nb::isinstance<LineRecombination>(h)) return std::make_unique<LineRecombination>(nb::cast<const LineRecombination&>(h));
    if (nb::isinstance<IntermediateRecombination>(h)) return std::make_unique<IntermediateRecombination>(nb::cast<const IntermediateRecombination&>(h));
    if (nb::isinstance<DifferentialEvolutionCrossover>(h)) return std::make_unique<DifferentialEvolutionCrossover>(nb::cast<const DifferentialEvolutionCrossover&>(h));
    if (nb::isinstance<UniformKVectorCrossover>(h)) return std::make_unique<UniformKVectorCrossover>(nb::cast<const UniformKVectorCrossover&>(h));
    if (nb::isinstance<OrderCrossover>(h)) return std::make_unique<OrderCrossover>(nb::cast<const OrderCrossover&>(h));
    if (nb::isinstance<PartiallyMappedCrossover>(h)) return std::make_unique<PartiallyMappedCrossover>(nb::cast<const PartiallyMappedCrossover&>(h));
    if (nb::isinstance<CycleCrossover>(h)) return std::make_unique<CycleCrossover>(nb::cast<const CycleCrossover&>(h));
    if (nb::isinstance<CutAndCrossfillCrossover>(h)) return std::make_unique<CutAndCrossfillCrossover>(nb::cast<const CutAndCrossfillCrossover&>(h));
    if (nb::isinstance<EdgeCrossover>(h)) return std::make_unique<EdgeCrossover>(nb::cast<const EdgeCrossover&>(h));
    if (nb::isinstance<DiploidRecombination>(h)) return std::make_unique<DiploidRecombination>(nb::cast<const DiploidRecombination&>(h));
    if (nb::isinstance<ga::fuzzy::FuzzyAdaptiveCrossover>(h)) return std::make_unique<ga::fuzzy::FuzzyAdaptiveCrossover>(nb::cast<const ga::fuzzy::FuzzyAdaptiveCrossover&>(h));
    return nullptr;
}

static inline std::unique_ptr<MutationOperator> cloneMutation(nb::handle h) {
    if (!h.is_valid() || h.is_none()) return nullptr;
    if (nb::isinstance<GaussianMutation>(h)) return std::make_unique<GaussianMutation>(nb::cast<const GaussianMutation&>(h));
    if (nb::isinstance<UniformMutation>(h)) return std::make_unique<UniformMutation>(nb::cast<const UniformMutation&>(h));
    if (nb::isinstance<BitFlipMutation>(h)) return std::make_unique<BitFlipMutation>(nb::cast<const BitFlipMutation&>(h));
    if (nb::isinstance<RandomResettingMutation>(h)) return std::make_unique<RandomResettingMutation>(nb::cast<const RandomResettingMutation&>(h));
    if (nb::isinstance<CreepMutation>(h)) return std::make_unique<CreepMutation>(nb::cast<const CreepMutation&>(h));
    if (nb::isinstance<SwapMutation>(h)) return std::make_unique<SwapMutation>(nb::cast<const SwapMutation&>(h));
    if (nb::isinstance<InversionMutation>(h)) return std::make_unique<InversionMutation>(nb::cast<const InversionMutation&>(h));
    if (nb::isinstance<InsertMutation>(h)) return std::make_unique<InsertMutation>(nb::cast<const InsertMutation&>(h));
    if (nb::isinstance<ScrambleMutation>(h)) return std::make_unique<ScrambleMutation>(nb::cast<const ScrambleMutation&>(h));
    if (nb::isinstance<ListMutation>(h)) return std::make_unique<ListMutation>(nb::cast<const ListMutation&>(h));
    if (nb::isinstance<SelfAdaptiveMutation>(h)) return std::make_unique<SelfAdaptiveMutation>(nb::cast<const SelfAdaptiveMutation&>(h));
    return nullptr;
}

NB_MODULE(_core, m) {
    m.doc() = "Genetic Algorithm framework — C++ core with Python bindings";

    // ------------------------------------------------------------------ Bounds
    nb::class_<ga::Bounds>(m, "Bounds", "Gene value bounds [lower, upper]")
        .def(nb::init<>())
        .def("__init__", [](ga::Bounds* b, double lo, double hi) {
            new (b) ga::Bounds{lo, hi};
        }, nb::arg("lower"), nb::arg("upper"))
        .def_rw("lower", &ga::Bounds::lower)
        .def_rw("upper", &ga::Bounds::upper)
        .def("__repr__", [](const ga::Bounds& b){
            return "Bounds(lower=" + std::to_string(b.lower) + ", upper=" + std::to_string(b.upper) + ")";
        });

    // ------------------------------------------------------------------ Config
    nb::class_<ga::Config>(m, "Config", "Genetic Algorithm configuration")
        .def(nb::init<>())
        .def_rw("population_size", &ga::Config::populationSize,  "Number of individuals")
        .def_rw("generations",     &ga::Config::generations,      "Number of generations")
        .def_rw("dimension",       &ga::Config::dimension,        "Gene vector length")
        .def_rw("crossover_rate",  &ga::Config::crossoverRate,    "Crossover probability [0,1]")
        .def_rw("mutation_rate",   &ga::Config::mutationRate,     "Per-gene mutation probability [0,1]")
        .def_rw("bounds",          &ga::Config::bounds,           "Gene search bounds")
        .def_rw("elite_ratio",     &ga::Config::eliteRatio,       "Elite fraction preserved each gen [0,1]")
        .def_rw("seed",            &ga::Config::seed,             "RNG seed (0 = random)")
        .def("__repr__", [](const ga::Config& c){
            return "<Config pop=" + std::to_string(c.populationSize)
                 + " gen=" + std::to_string(c.generations)
                 + " dim=" + std::to_string(c.dimension) + ">";
        });

    // ------------------------------------------------------------------ Result
    nb::class_<ga::Result>(m, "Result", "Results returned by GeneticAlgorithm.run()")
        .def(nb::init<>())
        .def_ro("best_genes",    &ga::Result::bestGenes,    "Best gene vector found")
        .def_ro("best_fitness",  &ga::Result::bestFitness,  "Fitness of the best individual")
        .def_ro("best_history",  &ga::Result::bestHistory,  "Best fitness per generation")
        .def_ro("avg_history",   &ga::Result::avgHistory,   "Average fitness per generation")
        .def_ro("evaluations",   &ga::Result::evaluations,  "Number of fitness evaluations")
        .def_ro("iterations",    &ga::Result::iterations,   "Number of completed generations")
        .def("__repr__", [](const ga::Result& r){
            return "<Result best_fitness=" + std::to_string(r.bestFitness) + ">";
        });

    nb::class_<ga::core::OptimizationResult>(m, "OptimizationResult", "Generic optimization result container")
        .def(nb::init<>())
        .def_rw("best_solution", &ga::core::OptimizationResult::bestSolution)
        .def_rw("best_fitness", &ga::core::OptimizationResult::bestFitness)
        .def_rw("best_history", &ga::core::OptimizationResult::bestHistory)
        .def_rw("avg_history", &ga::core::OptimizationResult::avgHistory)
        .def_rw("pareto_objectives", &ga::core::OptimizationResult::paretoObjectives)
        .def_rw("pareto_genes", &ga::core::OptimizationResult::paretoGenes)
        .def_rw("evaluations", &ga::core::OptimizationResult::evaluations)
        .def_rw("generations", &ga::core::OptimizationResult::generations);

    // ----------------------------------------------------------- Metaheuristics
    nb::class_<ga::metaheuristics::SearchConfig>(m, "SearchConfig")
        .def(nb::init<>())
        .def_rw("population_size", &ga::metaheuristics::SearchConfig::populationSize)
        .def_rw("iterations", &ga::metaheuristics::SearchConfig::iterations)
        .def_rw("dimension", &ga::metaheuristics::SearchConfig::dimension)
        .def_rw("bounds", &ga::metaheuristics::SearchConfig::bounds)
        .def_rw("seed", &ga::metaheuristics::SearchConfig::seed)
        .def_rw("threads", &ga::metaheuristics::SearchConfig::threads);

    nb::class_<ga::metaheuristics::ProgressState>(m, "ProgressState")
        .def(nb::init<>())
        .def_rw("iteration", &ga::metaheuristics::ProgressState::iteration)
        .def_rw("max_iterations", &ga::metaheuristics::ProgressState::maxIterations)
        .def_rw("normalized_diversity",
                       &ga::metaheuristics::ProgressState::normalizedDiversity)
        .def_rw("relative_improvement",
                       &ga::metaheuristics::ProgressState::relativeImprovement)
        .def_rw("stagnation", &ga::metaheuristics::ProgressState::stagnation);

    nb::class_<ga::metaheuristics::ControlSignal>(m, "ControlSignal")
        .def(nb::init<>())
        .def("__init__", [](ga::metaheuristics::ControlSignal* signal,
                            double exploration,
                            double exploitation,
                            double evaporation,
                            double randomization) {
                 new (signal) ga::metaheuristics::ControlSignal{exploration,
                                                                exploitation,
                                                                evaporation,
                                                                randomization};
             },
             nb::arg("exploration") = 1.0,
             nb::arg("exploitation") = 1.0,
             nb::arg("evaporation") = 1.0,
             nb::arg("randomization") = 1.0)
        .def_rw("exploration", &ga::metaheuristics::ControlSignal::exploration)
        .def_rw("exploitation", &ga::metaheuristics::ControlSignal::exploitation)
        .def_rw("evaporation", &ga::metaheuristics::ControlSignal::evaporation)
        .def_rw("randomization", &ga::metaheuristics::ControlSignal::randomization);

    nb::class_<ga::metaheuristics::IAdaptiveController,
               PyAdaptiveController>(
        m, "AdaptiveController", "Base class for optional adaptive controllers")
        .def(nb::init<>())
        .def("update", &ga::metaheuristics::IAdaptiveController::update,
             nb::arg("state"));

    nb::class_<ga::fuzzy::FuzzyControllerConfig>(m, "FuzzyControllerConfig")
        .def(nb::init<>())
        .def_rw("low_zero", &ga::fuzzy::FuzzyControllerConfig::lowZero)
        .def_rw("medium_center", &ga::fuzzy::FuzzyControllerConfig::mediumCenter)
        .def_rw("high_start", &ga::fuzzy::FuzzyControllerConfig::highStart)
        .def_rw("improvement_scale",
                       &ga::fuzzy::FuzzyControllerConfig::improvementScale)
        .def_rw("low_diversity_stagnant",
                       &ga::fuzzy::FuzzyControllerConfig::lowDiversityStagnant)
        .def_rw("low_diversity_slow",
                       &ga::fuzzy::FuzzyControllerConfig::lowDiversitySlow)
        .def_rw("balanced", &ga::fuzzy::FuzzyControllerConfig::balanced)
        .def_rw("diverse_productive",
                       &ga::fuzzy::FuzzyControllerConfig::diverseProductive)
        .def_rw("diverse_stagnant",
                       &ga::fuzzy::FuzzyControllerConfig::diverseStagnant);

    nb::class_<ga::fuzzy::FuzzyAdaptiveController,
               ga::metaheuristics::IAdaptiveController>(
        m, "FuzzyAdaptiveController")
        .def(nb::init<ga::fuzzy::FuzzyControllerConfig>(),
             nb::arg("config") = ga::fuzzy::FuzzyControllerConfig{})
        .def("update", &ga::fuzzy::FuzzyAdaptiveController::update,
             nb::arg("state"))
        .def_prop_ro("config", [](const ga::fuzzy::FuzzyAdaptiveController& self) {
            return self.config();
        });

    nb::class_<ga::metaheuristics::IContinuousOptimizer,
               PyContinuousOptimizer>(
        m, "ContinuousOptimizer", "Base class for continuous optimizers")
        .def(nb::init<>())
        .def("name", &ga::metaheuristics::IContinuousOptimizer::name)
        .def("optimize",
             [](ga::metaheuristics::IContinuousOptimizer& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 return optimizeFromPython(self, std::move(fitness), seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{});

    nb::enum_<ga::pso::PsoVariant>(m, "PsoVariant")
        .value("global_best", ga::pso::PsoVariant::GlobalBest)
        .value("local_best", ga::pso::PsoVariant::LocalBest)
        .value("constriction", ga::pso::PsoVariant::Constriction)
        .value("bare_bones", ga::pso::PsoVariant::BareBones)
        .value("fully_informed", ga::pso::PsoVariant::FullyInformed)
        .value("quantum_behaved", ga::pso::PsoVariant::QuantumBehaved)
        .value("binary", ga::pso::PsoVariant::Binary);

    nb::class_<ga::pso::PsoConfig>(m, "PsoConfig")
        .def(nb::init<>())
        .def_rw("search", &ga::pso::PsoConfig::search)
        .def_rw("variant", &ga::pso::PsoConfig::variant)
        .def_rw("inertia", &ga::pso::PsoConfig::inertia)
        .def_rw("cognitive", &ga::pso::PsoConfig::cognitive)
        .def_rw("social", &ga::pso::PsoConfig::social)
        .def_rw("constriction", &ga::pso::PsoConfig::constriction)
        .def_rw("velocity_clamp", &ga::pso::PsoConfig::velocityClamp)
        .def_rw("binary_velocity_clamp",
                       &ga::pso::PsoConfig::binaryVelocityClamp)
        .def_rw("neighborhood_radius",
                       &ga::pso::PsoConfig::neighborhoodRadius)
        .def_rw("quantum_beta", &ga::pso::PsoConfig::quantumBeta)
        .def_prop_rw("controller",
                      [](const ga::pso::PsoConfig& config) {
                          return mutableController(config.controller);
                      },
                      [](ga::pso::PsoConfig& config,
                         std::shared_ptr<ga::metaheuristics::IAdaptiveController> controller) {
                          config.controller = std::move(controller);
                      });

    nb::class_<ga::pso::ParticleSwarmOptimizer,
               ga::metaheuristics::IContinuousOptimizer>(
        m, "ParticleSwarmOptimizer")
        .def(nb::init<ga::pso::PsoConfig>(), nb::arg("config") = ga::pso::PsoConfig{})
        .def("name", &ga::pso::ParticleSwarmOptimizer::name)
        .def("optimize",
             [](ga::pso::ParticleSwarmOptimizer& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 return optimizeFromPython(self, std::move(fitness), seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{})
        .def_prop_ro("config", [](const ga::pso::ParticleSwarmOptimizer& self) {
            return self.config();
        });

    nb::class_<ga::aco::DenseGraph>(m, "DenseGraph")
        .def(nb::init<std::vector<std::vector<double>>, bool>(),
             nb::arg("costs"), nb::arg("symmetric") = true)
        .def_prop_ro("size", &ga::aco::DenseGraph::size)
        .def_prop_ro("symmetric", &ga::aco::DenseGraph::symmetric)
        .def("cost", &ga::aco::DenseGraph::cost, nb::arg("from_node"), nb::arg("to_node"));

    nb::enum_<ga::aco::AntColonyVariant>(m, "AntColonyVariant")
        .value("ant_system", ga::aco::AntColonyVariant::AntSystem)
        .value("elitist_ant_system", ga::aco::AntColonyVariant::ElitistAntSystem)
        .value("rank_based_ant_system", ga::aco::AntColonyVariant::RankBasedAntSystem)
        .value("ant_colony_system", ga::aco::AntColonyVariant::AntColonySystem)
        .value("max_min_ant_system", ga::aco::AntColonyVariant::MaxMinAntSystem);

    nb::class_<ga::aco::AntColonyConfig>(m, "AntColonyConfig")
        .def(nb::init<>())
        .def_rw("ants", &ga::aco::AntColonyConfig::ants)
        .def_rw("iterations", &ga::aco::AntColonyConfig::iterations)
        .def_rw("alpha", &ga::aco::AntColonyConfig::alpha)
        .def_rw("beta", &ga::aco::AntColonyConfig::beta)
        .def_rw("evaporation", &ga::aco::AntColonyConfig::evaporation)
        .def_rw("deposit_scale", &ga::aco::AntColonyConfig::depositScale)
        .def_rw("initial_pheromone", &ga::aco::AntColonyConfig::initialPheromone)
        .def_rw("elitist_weight", &ga::aco::AntColonyConfig::elitistWeight)
        .def_rw("rank_count", &ga::aco::AntColonyConfig::rankCount)
        .def_rw("exploitation_probability",
                       &ga::aco::AntColonyConfig::exploitationProbability)
        .def_rw("local_evaporation", &ga::aco::AntColonyConfig::localEvaporation)
        .def_rw("candidate_list_size", &ga::aco::AntColonyConfig::candidateListSize)
        .def_rw("min_pheromone", &ga::aco::AntColonyConfig::minPheromone)
        .def_rw("max_pheromone", &ga::aco::AntColonyConfig::maxPheromone)
        .def_rw("variant", &ga::aco::AntColonyConfig::variant)
        .def_rw("seed", &ga::aco::AntColonyConfig::seed)
        .def_prop_rw("controller",
                      [](const ga::aco::AntColonyConfig& config) {
                          return mutableController(config.controller);
                      },
                      [](ga::aco::AntColonyConfig& config,
                         std::shared_ptr<ga::metaheuristics::IAdaptiveController> controller) {
                          config.controller = std::move(controller);
                      });

    nb::class_<ga::aco::AntColonyResult>(m, "AntColonyResult")
        .def(nb::init<>())
        .def_ro("best_tour", &ga::aco::AntColonyResult::bestTour)
        .def_ro("best_cost", &ga::aco::AntColonyResult::bestCost)
        .def_ro("best_cost_history", &ga::aco::AntColonyResult::bestCostHistory)
        .def_ro("evaluations", &ga::aco::AntColonyResult::evaluations)
        .def_ro("iterations", &ga::aco::AntColonyResult::iterations);

    nb::class_<ga::aco::AntColonyOptimizer>(m, "AntColonyOptimizer")
        .def(nb::init<ga::aco::AntColonyConfig>(),
             nb::arg("config") = ga::aco::AntColonyConfig{})
        .def("solve", [](const ga::aco::AntColonyOptimizer& self,
                          const ga::aco::DenseGraph& graph) {
                 nb::gil_scoped_release release;
                 return self.solve(graph);
             }, nb::arg("graph"))
        .def_prop_ro("config", [](const ga::aco::AntColonyOptimizer& self) {
            return self.config();
        });

    nb::class_<ga::aco::AcorConfig>(m, "AcorConfig")
        .def(nb::init<>())
        .def_rw("search", &ga::aco::AcorConfig::search)
        .def_rw("archive_size", &ga::aco::AcorConfig::archiveSize)
        .def_rw("sample_count", &ga::aco::AcorConfig::sampleCount)
        .def_rw("locality", &ga::aco::AcorConfig::locality)
        .def_rw("convergence_speed", &ga::aco::AcorConfig::convergenceSpeed)
        .def_prop_rw("controller",
                      [](const ga::aco::AcorConfig& config) {
                          return mutableController(config.controller);
                      },
                      [](ga::aco::AcorConfig& config,
                         std::shared_ptr<ga::metaheuristics::IAdaptiveController> controller) {
                          config.controller = std::move(controller);
                      });

    nb::class_<ga::aco::ContinuousAntColonyOptimizer,
               ga::metaheuristics::IContinuousOptimizer>(
        m, "ContinuousAntColonyOptimizer")
        .def(nb::init<ga::aco::AcorConfig>(), nb::arg("config") = ga::aco::AcorConfig{})
        .def("name", &ga::aco::ContinuousAntColonyOptimizer::name)
        .def("optimize",
             [](ga::aco::ContinuousAntColonyOptimizer& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 return optimizeFromPython(self, std::move(fitness), seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{})
        .def_prop_ro("config",
            [](const ga::aco::ContinuousAntColonyOptimizer& self) {
                return self.config();
            });

    nb::class_<ga::gsa::GsaConfig>(m, "GsaConfig")
        .def(nb::init<>())
        .def_rw("search", &ga::gsa::GsaConfig::search)
        .def_rw("gravitational_constant", &ga::gsa::GsaConfig::gravitationalConstant)
        .def_rw("decay", &ga::gsa::GsaConfig::decay)
        .def_rw("epsilon", &ga::gsa::GsaConfig::epsilon)
        .def_rw("final_elite_fraction", &ga::gsa::GsaConfig::finalEliteFraction)
        .def_prop_rw("controller",
                      [](const ga::gsa::GsaConfig& config) {
                          return mutableController(config.controller);
                      },
                      [](ga::gsa::GsaConfig& config,
                         std::shared_ptr<ga::metaheuristics::IAdaptiveController> controller) {
                          config.controller = std::move(controller);
                      });

    nb::class_<ga::gsa::GravitationalSearchOptimizer,
               ga::metaheuristics::IContinuousOptimizer>(
        m, "GravitationalSearchOptimizer")
        .def(nb::init<ga::gsa::GsaConfig>(), nb::arg("config") = ga::gsa::GsaConfig{})
        .def("name", &ga::gsa::GravitationalSearchOptimizer::name)
        .def("optimize",
             [](ga::gsa::GravitationalSearchOptimizer& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 return optimizeFromPython(self, std::move(fitness), seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{})
        .def_prop_ro("config",
            [](const ga::gsa::GravitationalSearchOptimizer& self) {
                return self.config();
            });

    nb::class_<ga::fuzzy::FuzzyCMeansConfig>(m, "FuzzyCMeansConfig")
        .def(nb::init<>())
        .def_rw("clusters", &ga::fuzzy::FuzzyCMeansConfig::clusters)
        .def_rw("max_iterations", &ga::fuzzy::FuzzyCMeansConfig::maxIterations)
        .def_rw("fuzziness", &ga::fuzzy::FuzzyCMeansConfig::fuzziness)
        .def_rw("tolerance", &ga::fuzzy::FuzzyCMeansConfig::tolerance)
        .def_rw("seed", &ga::fuzzy::FuzzyCMeansConfig::seed);

    nb::class_<ga::fuzzy::FuzzyCMeansResult>(m, "FuzzyCMeansResult")
        .def(nb::init<>())
        .def_ro("centers", &ga::fuzzy::FuzzyCMeansResult::centers)
        .def_ro("membership", &ga::fuzzy::FuzzyCMeansResult::membership)
        .def_ro("objective_history", &ga::fuzzy::FuzzyCMeansResult::objectiveHistory)
        .def_ro("iterations", &ga::fuzzy::FuzzyCMeansResult::iterations)
        .def_ro("converged", &ga::fuzzy::FuzzyCMeansResult::converged)
        .def("labels", &ga::fuzzy::FuzzyCMeansResult::labels);

    nb::class_<ga::fuzzy::FuzzyCMeans>(m, "FuzzyCMeans")
        .def(nb::init<ga::fuzzy::FuzzyCMeansConfig>(),
             nb::arg("config") = ga::fuzzy::FuzzyCMeansConfig{})
        .def("fit", &ga::fuzzy::FuzzyCMeans::fit, nb::arg("data"),
             nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("config", [](const ga::fuzzy::FuzzyCMeans& self) {
            return self.config();
        });

    nb::class_<ga::metaheuristics::GeneticAlgorithmAdapter,
               ga::metaheuristics::IContinuousOptimizer>(
        m, "GeneticAlgorithmAdapter")
        .def(nb::init<ga::Config>(), nb::arg("config"))
        .def("name", &ga::metaheuristics::GeneticAlgorithmAdapter::name)
        .def("optimize",
             [](ga::metaheuristics::GeneticAlgorithmAdapter& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 return optimizeFromPython(self, std::move(fitness), seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{});

    nb::class_<ga::hybrid::StageResult>(m, "MetaheuristicStageResult")
        .def_ro("optimizer", &ga::hybrid::StageResult::optimizer)
        .def_ro("result", &ga::hybrid::StageResult::result);

    nb::class_<ga::hybrid::PipelineResult>(m, "MetaheuristicPipelineResult")
        .def_ro("combined", &ga::hybrid::PipelineResult::combined)
        .def_ro("stages", &ga::hybrid::PipelineResult::stages);

    nb::class_<ga::hybrid::MetaheuristicPipeline,
               ga::metaheuristics::IContinuousOptimizer>(
        m, "MetaheuristicPipeline")
        .def(nb::init<>())
        .def("add",
             [](ga::hybrid::MetaheuristicPipeline& self,
                 std::shared_ptr<ga::metaheuristics::IContinuousOptimizer> optimizer)
                 -> ga::hybrid::MetaheuristicPipeline& {
                 return self.addShared(std::move(optimizer));
             },
             nb::arg("optimizer"),
             nb::rv_policy::reference_internal)
        .def("name", &ga::hybrid::MetaheuristicPipeline::name)
        .def("size", &ga::hybrid::MetaheuristicPipeline::size)
        .def("optimize",
             [](ga::hybrid::MetaheuristicPipeline& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 return optimizeFromPython(self, std::move(fitness), seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{})
        .def("optimize_detailed",
             [](ga::hybrid::MetaheuristicPipeline& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 ga::Fitness wrapped = wrapPythonFitness(std::move(fitness));
                 nb::gil_scoped_release release;
                 return self.optimizeDetailed(wrapped, seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{});

    // ------------------------------------------------------------- CLPSO
    nb::class_<ga::pso::ClpsoConfig>(m, "ClpsoConfig")
        .def(nb::init<>())
        .def_rw("search", &ga::pso::ClpsoConfig::search)
        .def_rw("inertia", &ga::pso::ClpsoConfig::inertia)
        .def_rw("learning_rate", &ga::pso::ClpsoConfig::learningRate)
        .def_rw("velocity_clamp", &ga::pso::ClpsoConfig::velocityClamp)
        .def_rw("refresh_gap", &ga::pso::ClpsoConfig::refreshGap)
        .def_prop_rw("controller",
                      [](const ga::pso::ClpsoConfig& config) {
                          return mutableController(config.controller);
                      },
                      [](ga::pso::ClpsoConfig& config,
                         std::shared_ptr<ga::metaheuristics::IAdaptiveController> controller) {
                          config.controller = std::move(controller);
                      });

    nb::class_<ga::pso::ComprehensiveLearningPso,
               ga::metaheuristics::IContinuousOptimizer>(m, "ComprehensiveLearningPso")
        .def(nb::init<ga::pso::ClpsoConfig>(), nb::arg("config") = ga::pso::ClpsoConfig{})
        .def("name", &ga::pso::ComprehensiveLearningPso::name)
        .def("optimize",
             [](ga::pso::ComprehensiveLearningPso& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 return optimizeFromPython(self, std::move(fitness), seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{})
        .def_prop_ro("config", [](const ga::pso::ComprehensiveLearningPso& self) {
            return self.config();
        });

    // --------------------------------------------------------------- GSO
    nb::enum_<ga::gso::GsoVariant>(m, "GsoVariant")
        .value("standard", ga::gso::GsoVariant::Standard)
        .value("adaptive_step", ga::gso::GsoVariant::AdaptiveStep)
        .value("levy_flight", ga::gso::GsoVariant::LevyFlight)
        .value("multi_modal", ga::gso::GsoVariant::MultiModal);

    nb::class_<ga::gso::GsoConfig>(m, "GsoConfig")
        .def(nb::init<>())
        .def_rw("search", &ga::gso::GsoConfig::search)
        .def_rw("variant", &ga::gso::GsoConfig::variant)
        .def_rw("rho", &ga::gso::GsoConfig::rho)
        .def_rw("gamma", &ga::gso::GsoConfig::gamma)
        .def_rw("beta", &ga::gso::GsoConfig::beta)
        .def_rw("step_size", &ga::gso::GsoConfig::stepSize)
        .def_rw("min_step_size", &ga::gso::GsoConfig::minStepSize)
        .def_rw("max_step_size", &ga::gso::GsoConfig::maxStepSize)
        .def_rw("sensor_range", &ga::gso::GsoConfig::sensorRange)
        .def_rw("initial_sensor_range", &ga::gso::GsoConfig::initialSensorRange)
        .def_rw("target_neighbors", &ga::gso::GsoConfig::targetNeighbors)
        .def_rw("initial_luciferin", &ga::gso::GsoConfig::initialLuciferin)
        .def_rw("niche_radius", &ga::gso::GsoConfig::nicheRadius)
        .def_prop_rw("controller",
                      [](const ga::gso::GsoConfig& config) {
                          return mutableController(config.controller);
                      },
                      [](ga::gso::GsoConfig& config,
                         std::shared_ptr<ga::metaheuristics::IAdaptiveController> controller) {
                          config.controller = std::move(controller);
                      });

    nb::class_<ga::gso::GlowwormSwarmOptimizer,
               ga::metaheuristics::IContinuousOptimizer>(m, "GlowwormSwarmOptimizer")
        .def(nb::init<ga::gso::GsoConfig>(), nb::arg("config") = ga::gso::GsoConfig{})
        .def("name", &ga::gso::GlowwormSwarmOptimizer::name)
        .def("optimize",
             [](ga::gso::GlowwormSwarmOptimizer& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 return optimizeFromPython(self, std::move(fitness), seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{})
        .def_prop_ro("config", [](const ga::gso::GlowwormSwarmOptimizer& self) {
            return self.config();
        });

    // ------------------------------------------------------------- Shake
    nb::enum_<ga::shake::ShakeType>(m, "ShakeType")
        .value("uniform", ga::shake::ShakeType::Uniform)
        .value("gaussian", ga::shake::ShakeType::Gaussian)
        .value("cauchy", ga::shake::ShakeType::Cauchy)
        .value("levy_flight", ga::shake::ShakeType::LevyFlight)
        .value("opposition", ga::shake::ShakeType::Opposition)
        .value("partial_dimension", ga::shake::ShakeType::PartialDimension);

    nb::class_<ga::shake::ShakeConfig>(m, "ShakeConfig")
        .def(nb::init<>())
        .def_rw("type", &ga::shake::ShakeConfig::type)
        .def_rw("intensity", &ga::shake::ShakeConfig::intensity)
        .def_rw("probability", &ga::shake::ShakeConfig::probability)
        .def_rw("levy_beta", &ga::shake::ShakeConfig::levyBeta)
        .def_rw("partial_dims", &ga::shake::ShakeConfig::partialDims)
        .def_rw("retain_best", &ga::shake::ShakeConfig::retainBest);

    nb::class_<ga::shake::AdaptiveShakerConfig>(m, "AdaptiveShakerConfig")
        .def(nb::init<>())
        .def_rw("shake", &ga::shake::AdaptiveShakerConfig::shake)
        .def_rw("stagnation_threshold", &ga::shake::AdaptiveShakerConfig::stagnationThreshold)
        .def_rw("diversity_threshold", &ga::shake::AdaptiveShakerConfig::diversityThreshold)
        .def_rw("fraction_to_shake", &ga::shake::AdaptiveShakerConfig::fractionToShake);

    nb::class_<ga::shake::AdaptiveSwarmShaker>(m, "AdaptiveSwarmShaker")
        .def(nb::init<ga::shake::AdaptiveShakerConfig>(), nb::arg("config") = ga::shake::AdaptiveShakerConfig{})
        .def("reset", &ga::shake::AdaptiveSwarmShaker::reset)
        .def_prop_ro("config", &ga::shake::AdaptiveSwarmShaker::config)
        .def_prop_ro("stagnation_count", &ga::shake::AdaptiveSwarmShaker::stagnationCount)
        .def_prop_ro("last_diversity", &ga::shake::AdaptiveSwarmShaker::lastDiversity);

    m.def("apply_shake", [](std::vector<double> solution, const ga::Bounds& bounds, const ga::shake::ShakeConfig& cfg, unsigned int seed) {
        std::mt19937 rng(seed);
        ga::shake::applyShake(solution, bounds, cfg, rng);
        return solution;
    }, nb::arg("solution"), nb::arg("bounds"), nb::arg("config") = ga::shake::ShakeConfig{}, nb::arg("seed") = 42);

    // ------------------------------------------------------------- Fuzzy
    nb::class_<ga::fuzzy::LinguisticVariable>(m, "LinguisticVariable")
        .def("__init__", [](ga::fuzzy::LinguisticVariable* lv, std::string name, double minVal, double maxVal) {
            new (lv) ga::fuzzy::LinguisticVariable(std::move(name), ga::fuzzy::Interval{minVal, maxVal});
        }, nb::arg("name"), nb::arg("min_val"), nb::arg("max_val"))
        .def("name", &ga::fuzzy::LinguisticVariable::name)
        .def("add_triangular", [](ga::fuzzy::LinguisticVariable& self, const std::string& term, double a, double b, double c) {
            self.addTerm(term, ga::fuzzy::makeTriangularMF(a, b, c));
        }, nb::arg("term"), nb::arg("a"), nb::arg("b"), nb::arg("c"))
        .def("add_trapezoidal", [](ga::fuzzy::LinguisticVariable& self, const std::string& term, double a, double b, double c, double d) {
            self.addTerm(term, ga::fuzzy::makeTrapezoidalMF(a, b, c, d));
        }, nb::arg("term"), nb::arg("a"), nb::arg("b"), nb::arg("c"), nb::arg("d"))
        .def("add_gaussian", [](ga::fuzzy::LinguisticVariable& self, const std::string& term, double mean, double sigma) {
            self.addTerm(term, ga::fuzzy::makeGaussianMF(mean, sigma));
        }, nb::arg("term"), nb::arg("mean"), nb::arg("sigma"))
        .def("fuzzify", &ga::fuzzy::LinguisticVariable::fuzzify, nb::arg("crisp_value"))
        .def("evaluate_term", &ga::fuzzy::LinguisticVariable::evaluateTerm, nb::arg("term"), nb::arg("crisp_value"));

    nb::class_<ga::fuzzy::MamdaniSystem>(m, "MamdaniSystem")
        .def(nb::init<>())
        .def("add_input", &ga::fuzzy::MamdaniSystem::addInput, nb::arg("name"), nb::arg("min_val"), nb::arg("max_val"))
        .def("add_input_triangular", [](ga::fuzzy::MamdaniSystem& self, const std::string& varName, const std::string& term, double a, double b, double c) {
            self.addInputTerm(varName, term, ga::fuzzy::makeTriangularMF(a, b, c));
        }, nb::arg("var_name"), nb::arg("term"), nb::arg("a"), nb::arg("b"), nb::arg("c"))
        .def("add_input_trapezoidal", [](ga::fuzzy::MamdaniSystem& self, const std::string& varName, const std::string& term, double a, double b, double c, double d) {
            self.addInputTerm(varName, term, ga::fuzzy::makeTrapezoidalMF(a, b, c, d));
        }, nb::arg("var_name"), nb::arg("term"), nb::arg("a"), nb::arg("b"), nb::arg("c"), nb::arg("d"))
        .def("add_input_gaussian", [](ga::fuzzy::MamdaniSystem& self, const std::string& varName, const std::string& term, double mean, double sigma) {
            self.addInputTerm(varName, term, ga::fuzzy::makeGaussianMF(mean, sigma));
        }, nb::arg("var_name"), nb::arg("term"), nb::arg("mean"), nb::arg("sigma"))
        .def("add_output", &ga::fuzzy::MamdaniSystem::addOutput, nb::arg("name"), nb::arg("min_val"), nb::arg("max_val"))
        .def("add_output_triangular", [](ga::fuzzy::MamdaniSystem& self, const std::string& varName, const std::string& term, double a, double b, double c) {
            self.addOutputTerm(varName, term, ga::fuzzy::makeTriangularMF(a, b, c));
        }, nb::arg("var_name"), nb::arg("term"), nb::arg("a"), nb::arg("b"), nb::arg("c"))
        .def("add_output_trapezoidal", [](ga::fuzzy::MamdaniSystem& self, const std::string& varName, const std::string& term, double a, double b, double c, double d) {
            self.addOutputTerm(varName, term, ga::fuzzy::makeTrapezoidalMF(a, b, c, d));
        }, nb::arg("var_name"), nb::arg("term"), nb::arg("a"), nb::arg("b"), nb::arg("c"), nb::arg("d"))
        .def("add_rule", nb::overload_cast<const std::string&>(&ga::fuzzy::MamdaniSystem::addRule), nb::arg("rule_string"))
        .def("evaluate", &ga::fuzzy::MamdaniSystem::evaluate, nb::arg("inputs"))
        .def("evaluate_single", &ga::fuzzy::MamdaniSystem::evaluateSingle, nb::arg("output_var"), nb::arg("inputs"), nb::arg("fallback") = 0.0);

    nb::class_<ga::fuzzy::SugenoSystem>(m, "SugenoSystem")
        .def(nb::init<>())
        .def("add_input", &ga::fuzzy::SugenoSystem::addInput, nb::arg("name"), nb::arg("min_val"), nb::arg("max_val"))
        .def("add_input_triangular", [](ga::fuzzy::SugenoSystem& self, const std::string& varName, const std::string& term, double a, double b, double c) {
            self.addInputTerm(varName, term, ga::fuzzy::makeTriangularMF(a, b, c));
        }, nb::arg("var_name"), nb::arg("term"), nb::arg("a"), nb::arg("b"), nb::arg("c"))
        .def("add_output", &ga::fuzzy::SugenoSystem::addOutput, nb::arg("name"), nb::arg("fallback") = 0.0)
        .def("evaluate", &ga::fuzzy::SugenoSystem::evaluate, nb::arg("inputs"))
        .def("evaluate_single", &ga::fuzzy::SugenoSystem::evaluateSingle, nb::arg("output_var"), nb::arg("inputs"), nb::arg("fallback") = 0.0);

    nb::enum_<ga::fuzzy::FuzzyControllerBackend>(m, "FuzzyControllerBackend")
        .value("mamdani", ga::fuzzy::FuzzyControllerBackend::Mamdani)
        .value("sugeno", ga::fuzzy::FuzzyControllerBackend::Sugeno);

    nb::class_<ga::fuzzy::FuzzyAlgorithmAdapterConfig>(m, "FuzzyAlgorithmAdapterConfig")
        .def(nb::init<>())
        .def_rw("backend", &ga::fuzzy::FuzzyAlgorithmAdapterConfig::backend)
        .def_rw("improvement_scale", &ga::fuzzy::FuzzyAlgorithmAdapterConfig::improvementScale)
        .def_rw("enable_fcm", &ga::fuzzy::FuzzyAlgorithmAdapterConfig::enableFcm)
        .def_rw("fcm_clusters", &ga::fuzzy::FuzzyAlgorithmAdapterConfig::fcmClusters);

    nb::class_<ga::fuzzy::FuzzyAlgorithmAdapter,
               ga::metaheuristics::IAdaptiveController>(m, "FuzzyAlgorithmAdapter")
        .def(nb::init<ga::fuzzy::FuzzyAlgorithmAdapterConfig>(),
             nb::arg("config") = ga::fuzzy::FuzzyAlgorithmAdapterConfig{})
        .def(nb::init<ga::fuzzy::MamdaniSystem>(), nb::arg("mamdani"))
        .def(nb::init<ga::fuzzy::SugenoSystem>(), nb::arg("sugeno"))
        .def("update", &ga::fuzzy::FuzzyAlgorithmAdapter::update, nb::arg("state"))
        .def("last_shake_intensity", &ga::fuzzy::FuzzyAlgorithmAdapter::lastShakeIntensity)
        .def_prop_ro("config", &ga::fuzzy::FuzzyAlgorithmAdapter::config);

    // ----------------------------------------------------------- Hybrids
    nb::class_<ga::hybrid::InterleavedHybridConfig>(m, "InterleavedHybridConfig")
        .def(nb::init<>())
        .def_rw("search", &ga::hybrid::InterleavedHybridConfig::search)
        .def_rw("epochs", &ga::hybrid::InterleavedHybridConfig::epochs)
        .def_rw("stage1_iterations", &ga::hybrid::InterleavedHybridConfig::stage1Iterations)
        .def_rw("stage2_iterations", &ga::hybrid::InterleavedHybridConfig::stage2Iterations)
        .def_rw("migrated_elite_count", &ga::hybrid::InterleavedHybridConfig::migratedEliteCount);

    nb::class_<ga::hybrid::InterleavedHybridOptimizer,
               ga::metaheuristics::IContinuousOptimizer>(m, "InterleavedHybridOptimizer")
        .def(nb::init<std::shared_ptr<ga::metaheuristics::IContinuousOptimizer>,
                      std::shared_ptr<ga::metaheuristics::IContinuousOptimizer>,
                      ga::hybrid::InterleavedHybridConfig>(),
             nb::arg("stage1"), nb::arg("stage2"), nb::arg("config") = ga::hybrid::InterleavedHybridConfig{})
        .def("name", &ga::hybrid::InterleavedHybridOptimizer::name)
        .def("optimize",
             [](ga::hybrid::InterleavedHybridOptimizer& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 return optimizeFromPython(self, std::move(fitness), seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{});

    nb::class_<ga::hybrid::SwarmGeneticHybridConfig>(m, "SwarmGeneticHybridConfig")
        .def(nb::init<>())
        .def_rw("search", &ga::hybrid::SwarmGeneticHybridConfig::search)
        .def_rw("crossover_probability", &ga::hybrid::SwarmGeneticHybridConfig::crossoverProbability)
        .def_rw("mutation_probability", &ga::hybrid::SwarmGeneticHybridConfig::mutationProbability)
        .def_rw("inertia", &ga::hybrid::SwarmGeneticHybridConfig::inertia)
        .def_rw("cognitive", &ga::hybrid::SwarmGeneticHybridConfig::cognitive)
        .def_rw("social", &ga::hybrid::SwarmGeneticHybridConfig::social)
        .def_rw("shake", &ga::hybrid::SwarmGeneticHybridConfig::shake);

    nb::class_<ga::hybrid::SwarmGeneticHybridOptimizer,
               ga::metaheuristics::IContinuousOptimizer>(m, "SwarmGeneticHybridOptimizer")
        .def(nb::init<ga::hybrid::SwarmGeneticHybridConfig>(),
             nb::arg("config") = ga::hybrid::SwarmGeneticHybridConfig{})
        .def("name", &ga::hybrid::SwarmGeneticHybridOptimizer::name)
        .def("optimize",
             [](ga::hybrid::SwarmGeneticHybridOptimizer& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 return optimizeFromPython(self, std::move(fitness), seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{})
        .def_prop_ro("config", [](const ga::hybrid::SwarmGeneticHybridOptimizer& self) {
            return self.config();
        });

    // ---------------------------------------------------- Island Model (Distributed GA)
    nb::enum_<ga::algorithms::IslandTopology>(m, "IslandTopology")
        .value("Ring", ga::algorithms::IslandTopology::Ring)
        .value("Star", ga::algorithms::IslandTopology::Star)
        .value("FullyConnected", ga::algorithms::IslandTopology::FullyConnected)
        .value("RandomMesh", ga::algorithms::IslandTopology::RandomMesh);

    nb::enum_<ga::algorithms::MigrationPolicy>(m, "MigrationPolicy")
        .value("BestToWorst", ga::algorithms::MigrationPolicy::BestToWorst)
        .value("RandomToWorst", ga::algorithms::MigrationPolicy::RandomToWorst)
        .value("BestToRandom", ga::algorithms::MigrationPolicy::BestToRandom);

    nb::class_<ga::algorithms::IslandConfig>(m, "IslandConfig")
        .def(nb::init<>())
        .def_rw("num_islands", &ga::algorithms::IslandConfig::numIslands)
        .def_rw("island_population_size", &ga::algorithms::IslandConfig::islandPopulationSize)
        .def_rw("iterations", &ga::algorithms::IslandConfig::iterations)
        .def_rw("migration_interval", &ga::algorithms::IslandConfig::migrationInterval)
        .def_rw("migrants_per_exchange", &ga::algorithms::IslandConfig::migrantsPerExchange)
        .def_rw("topology", &ga::algorithms::IslandConfig::topology)
        .def_rw("policy", &ga::algorithms::IslandConfig::policy)
        .def_rw("search", &ga::algorithms::IslandConfig::search)
        .def_rw("crossover_rate", &ga::algorithms::IslandConfig::crossoverRate)
        .def_rw("mutation_rate", &ga::algorithms::IslandConfig::mutationRate)
        .def_rw("elite_ratio", &ga::algorithms::IslandConfig::eliteRatio);

    nb::class_<ga::algorithms::IslandModelOptimizer,
               ga::metaheuristics::IContinuousOptimizer>(m, "IslandModelOptimizer")
        .def(nb::init<ga::algorithms::IslandConfig>(),
             nb::arg("config") = ga::algorithms::IslandConfig{})
        .def("name", &ga::algorithms::IslandModelOptimizer::name)
        .def("optimize",
             [](ga::algorithms::IslandModelOptimizer& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 return optimizeFromPython(self, std::move(fitness), seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{});

    // ---------------------------------------------------- Real-Time Simpler Heuristics
    nb::enum_<ga::metaheuristics::CoolingSchedule>(m, "CoolingSchedule")
        .value("Geometric", ga::metaheuristics::CoolingSchedule::Geometric)
        .value("Linear", ga::metaheuristics::CoolingSchedule::Linear)
        .value("Logarithmic", ga::metaheuristics::CoolingSchedule::Logarithmic);

    nb::class_<ga::metaheuristics::SimulatedAnnealingConfig>(m, "SimulatedAnnealingConfig")
        .def(nb::init<>())
        .def_rw("search", &ga::metaheuristics::SimulatedAnnealingConfig::search)
        .def_rw("initial_temperature", &ga::metaheuristics::SimulatedAnnealingConfig::initialTemperature)
        .def_rw("min_temperature", &ga::metaheuristics::SimulatedAnnealingConfig::minTemperature)
        .def_rw("cooling_rate", &ga::metaheuristics::SimulatedAnnealingConfig::coolingRate)
        .def_rw("step_size", &ga::metaheuristics::SimulatedAnnealingConfig::stepSize)
        .def_rw("steps_per_temperature", &ga::metaheuristics::SimulatedAnnealingConfig::stepsPerTemperature)
        .def_rw("schedule", &ga::metaheuristics::SimulatedAnnealingConfig::schedule);

    nb::class_<ga::metaheuristics::SimulatedAnnealingOptimizer,
               ga::metaheuristics::IContinuousOptimizer>(m, "SimulatedAnnealingOptimizer")
        .def(nb::init<ga::metaheuristics::SimulatedAnnealingConfig>(),
             nb::arg("config") = ga::metaheuristics::SimulatedAnnealingConfig{})
        .def("name", &ga::metaheuristics::SimulatedAnnealingOptimizer::name)
        .def("optimize",
             [](ga::metaheuristics::SimulatedAnnealingOptimizer& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 return optimizeFromPython(self, std::move(fitness), seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{});

    nb::class_<ga::metaheuristics::HillClimbingConfig>(m, "HillClimbingConfig")
        .def(nb::init<>())
        .def_rw("search", &ga::metaheuristics::HillClimbingConfig::search)
        .def_rw("step_size", &ga::metaheuristics::HillClimbingConfig::stepSize)
        .def_rw("num_neighbors", &ga::metaheuristics::HillClimbingConfig::numNeighbors)
        .def_rw("restarts", &ga::metaheuristics::HillClimbingConfig::restarts)
        .def_rw("steepest_ascent", &ga::metaheuristics::HillClimbingConfig::steepestAscent);

    nb::class_<ga::metaheuristics::HillClimbingOptimizer,
               ga::metaheuristics::IContinuousOptimizer>(m, "HillClimbingOptimizer")
        .def(nb::init<ga::metaheuristics::HillClimbingConfig>(),
             nb::arg("config") = ga::metaheuristics::HillClimbingConfig{})
        .def("name", &ga::metaheuristics::HillClimbingOptimizer::name)
        .def("optimize",
             [](ga::metaheuristics::HillClimbingOptimizer& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 return optimizeFromPython(self, std::move(fitness), seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{});

    // ---------------------------------------------------- Dynamic Problems & Adaptive GA
    nb::enum_<ga::adaptive::DynamicStrategy>(m, "DynamicStrategy")
        .value("Hypermutation", ga::adaptive::DynamicStrategy::Hypermutation)
        .value("RandomImmigrants", ga::adaptive::DynamicStrategy::RandomImmigrants)
        .value("MemoryArchive", ga::adaptive::DynamicStrategy::MemoryArchive)
        .value("Hybrid", ga::adaptive::DynamicStrategy::Hybrid);

    nb::class_<ga::adaptive::DynamicGAConfig>(m, "DynamicGAConfig")
        .def(nb::init<>())
        .def_rw("search", &ga::adaptive::DynamicGAConfig::search)
        .def_rw("stagnation_window", &ga::adaptive::DynamicGAConfig::stagnationWindow)
        .def_rw("stagnation_tolerance", &ga::adaptive::DynamicGAConfig::stagnationTolerance)
        .def_rw("strategy", &ga::adaptive::DynamicGAConfig::strategy)
        .def_rw("immigrant_ratio", &ga::adaptive::DynamicGAConfig::immigrantRatio)
        .def_rw("hypermutation_factor", &ga::adaptive::DynamicGAConfig::hypermutationFactor)
        .def_rw("hypermutation_duration", &ga::adaptive::DynamicGAConfig::hypermutationDuration)
        .def_rw("diversity_threshold", &ga::adaptive::DynamicGAConfig::diversityThreshold)
        .def_rw("detect_environment_change", &ga::adaptive::DynamicGAConfig::detectEnvironmentChange)
        .def_rw("memory_size", &ga::adaptive::DynamicGAConfig::memorySize)
        .def_rw("base_crossover_rate", &ga::adaptive::DynamicGAConfig::baseCrossoverRate)
        .def_rw("base_mutation_rate", &ga::adaptive::DynamicGAConfig::baseMutationRate);

    nb::class_<ga::adaptive::DynamicGAOptimizer,
               ga::metaheuristics::IContinuousOptimizer>(m, "DynamicGAOptimizer")
        .def(nb::init<ga::adaptive::DynamicGAConfig>(),
             nb::arg("config") = ga::adaptive::DynamicGAConfig{})
        .def("name", &ga::adaptive::DynamicGAOptimizer::name)
        .def("optimize",
             [](ga::adaptive::DynamicGAOptimizer& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 return optimizeFromPython(self, std::move(fitness), seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{});

    // ---------------------------------------------------- High-level runners
    m.def("run_pso", [](nb::callable fitness,
                        ga::pso::PsoVariant variant,
                        std::size_t dimension,
                        std::size_t iterations,
                        ga::Bounds bounds,
                        unsigned seed,
                        std::size_t population_size) {
        ga::Fitness wrapped = wrapPythonFitness(std::move(fitness));
        nb::gil_scoped_release release;
        return ga::api::runPso(wrapped, variant, dimension, iterations, bounds, seed, population_size);
    }, nb::arg("fitness"),
       nb::arg("variant") = ga::pso::PsoVariant::GlobalBest,
       nb::arg("dimension") = 10,
       nb::arg("iterations") = 100,
       nb::arg("bounds") = ga::Bounds{-5.12, 5.12},
       nb::arg("seed") = 0,
       nb::arg("population_size") = 50);

    m.def("run_clpso", [](nb::callable fitness,
                          std::size_t dimension,
                          std::size_t iterations,
                          ga::Bounds bounds,
                          unsigned seed,
                          std::size_t population_size) {
        ga::Fitness wrapped = wrapPythonFitness(std::move(fitness));
        nb::gil_scoped_release release;
        return ga::api::runClpso(wrapped, dimension, iterations, bounds, seed, population_size);
    }, nb::arg("fitness"),
       nb::arg("dimension") = 10,
       nb::arg("iterations") = 100,
       nb::arg("bounds") = ga::Bounds{-5.12, 5.12},
       nb::arg("seed") = 0,
       nb::arg("population_size") = 50);

    m.def("run_gso", [](nb::callable fitness,
                        ga::gso::GsoVariant variant,
                        std::size_t dimension,
                        std::size_t iterations,
                        ga::Bounds bounds,
                        unsigned seed,
                        std::size_t population_size) {
        ga::Fitness wrapped = wrapPythonFitness(std::move(fitness));
        nb::gil_scoped_release release;
        return ga::api::runGso(wrapped, variant, dimension, iterations, bounds, seed, population_size);
    }, nb::arg("fitness"),
       nb::arg("variant") = ga::gso::GsoVariant::Standard,
       nb::arg("dimension") = 10,
       nb::arg("iterations") = 100,
       nb::arg("bounds") = ga::Bounds{-5.12, 5.12},
       nb::arg("seed") = 0,
       nb::arg("population_size") = 50);

    m.def("run_continuous_aco", [](nb::callable fitness,
                                   std::size_t dimension,
                                   std::size_t iterations,
                                   ga::Bounds bounds,
                                   unsigned seed,
                                   std::size_t archive_size) {
        ga::Fitness wrapped = wrapPythonFitness(std::move(fitness));
        nb::gil_scoped_release release;
        return ga::api::runContinuousAco(wrapped, dimension, iterations, bounds, seed, archive_size);
    }, nb::arg("fitness"),
       nb::arg("dimension") = 10,
       nb::arg("iterations") = 100,
       nb::arg("bounds") = ga::Bounds{-5.12, 5.12},
       nb::arg("seed") = 0,
       nb::arg("archive_size") = 50);

    m.def("run_graph_aco", [](const ga::aco::DenseGraph& graph,
                              ga::aco::AntColonyVariant variant,
                              std::size_t ants,
                              std::size_t iterations,
                              unsigned seed) {
        nb::gil_scoped_release release;
        return ga::api::runGraphAco(graph, variant, ants, iterations, seed);
    }, nb::arg("graph"),
       nb::arg("variant") = ga::aco::AntColonyVariant::AntSystem,
       nb::arg("ants") = 30,
       nb::arg("iterations") = 100,
       nb::arg("seed") = 0);

    m.def("run_island_model", [](nb::callable fitness,
                                 std::size_t num_islands,
                                 std::size_t island_population_size,
                                 std::size_t dimension,
                                 std::size_t iterations,
                                 ga::Bounds bounds,
                                 unsigned seed,
                                 ga::algorithms::IslandTopology topology) {
        ga::Fitness wrapped = wrapPythonFitness(std::move(fitness));
        nb::gil_scoped_release release;
        return ga::api::runIslandModel(wrapped, num_islands, island_population_size, dimension, iterations, bounds, seed, topology);
    }, nb::arg("fitness"),
       nb::arg("num_islands") = 4,
       nb::arg("island_population_size") = 50,
       nb::arg("dimension") = 10,
       nb::arg("iterations") = 100,
       nb::arg("bounds") = ga::Bounds{-5.12, 5.12},
       nb::arg("seed") = 0,
       nb::arg("topology") = ga::algorithms::IslandTopology::Ring);

    m.def("run_simulated_annealing", [](nb::callable fitness,
                                        std::size_t dimension,
                                        std::size_t iterations,
                                        ga::Bounds bounds,
                                        double initial_temperature,
                                        double cooling_rate,
                                        unsigned seed) {
        ga::Fitness wrapped = wrapPythonFitness(std::move(fitness));
        nb::gil_scoped_release release;
        return ga::api::runSimulatedAnnealing(wrapped, dimension, iterations, bounds, initial_temperature, cooling_rate, seed);
    }, nb::arg("fitness"),
       nb::arg("dimension") = 10,
       nb::arg("iterations") = 200,
       nb::arg("bounds") = ga::Bounds{-5.12, 5.12},
       nb::arg("initial_temperature") = 100.0,
       nb::arg("cooling_rate") = 0.95,
       nb::arg("seed") = 0);

    m.def("run_hill_climbing", [](nb::callable fitness,
                                  std::size_t dimension,
                                  std::size_t iterations,
                                  ga::Bounds bounds,
                                  double step_size,
                                  std::size_t restarts,
                                  unsigned seed) {
        ga::Fitness wrapped = wrapPythonFitness(std::move(fitness));
        nb::gil_scoped_release release;
        return ga::api::runHillClimbing(wrapped, dimension, iterations, bounds, step_size, restarts, seed);
    }, nb::arg("fitness"),
       nb::arg("dimension") = 10,
       nb::arg("iterations") = 200,
       nb::arg("bounds") = ga::Bounds{-5.12, 5.12},
       nb::arg("step_size") = 0.02,
       nb::arg("restarts") = 2,
       nb::arg("seed") = 0);

    m.def("run_dynamic_ga", [](nb::callable fitness,
                               std::size_t dimension,
                               std::size_t iterations,
                               ga::Bounds bounds,
                               std::size_t population_size,
                               unsigned seed,
                               ga::adaptive::DynamicStrategy strategy) {
        ga::Fitness wrapped = wrapPythonFitness(std::move(fitness));
        nb::gil_scoped_release release;
        return ga::api::runDynamicGa(wrapped, dimension, iterations, bounds, population_size, seed, strategy);
    }, nb::arg("fitness"),
       nb::arg("dimension") = 10,
       nb::arg("iterations") = 100,
       nb::arg("bounds") = ga::Bounds{-5.12, 5.12},
       nb::arg("population_size") = 50,
       nb::arg("seed") = 0,
       nb::arg("strategy") = ga::adaptive::DynamicStrategy::Hybrid);

    // ---------------------------------------------------------- Core abstractions
    nb::class_<ga::Evaluation>(m, "Evaluation", "Evaluation/objective record")
        .def(nb::init<>())
        .def_rw("objectives", &ga::Evaluation::objectives)
        .def_rw("feasible", &ga::Evaluation::feasible)
        .def_rw("penalty", &ga::Evaluation::penalty);

    nb::class_<ga::Individual>(m, "Individual", "Generic individual container")
        .def(nb::init<>())
        .def_rw("evaluation", &ga::Individual::evaluation)
        .def_rw("age", &ga::Individual::age);

    nb::class_<ga::IGenome>(m, "IGenome", "Genome interface")
        .def("encoding_name", &ga::IGenome::encodingName);

    // ------------------------------------------------------------- Representations
    nb::class_<ga::representations::VectorGenome<double>, ga::IGenome>(m, "VectorGenome")
        .def(nb::init<>())
        .def(nb::init<std::vector<double>>(), nb::arg("genes"))
        .def_rw("genes", &ga::representations::VectorGenome<double>::genes)
        .def("encoding_name", &ga::representations::VectorGenome<double>::encodingName);

    nb::class_<ga::representations::VectorGenome<int>, ga::IGenome>(m, "VectorGenomeInt")
        .def(nb::init<>())
        .def(nb::init<std::vector<int>>(), nb::arg("genes"))
        .def_rw("genes", &ga::representations::VectorGenome<int>::genes)
        .def("encoding_name", &ga::representations::VectorGenome<int>::encodingName);

    nb::class_<ga::representations::BitsetGenome, ga::IGenome>(m, "BitsetGenome")
        .def(nb::init<>())
        .def(nb::init<std::size_t, bool>(), nb::arg("size"), nb::arg("fill") = false)
        .def(nb::init<std::vector<bool>>(), nb::arg("bits"))
        .def_rw("bits", &ga::representations::BitsetGenome::bits)
        .def("size", &ga::representations::BitsetGenome::size)
        .def("hamming_distance", &ga::representations::BitsetGenome::hammingDistance, nb::arg("other"))
        .def("popcount", &ga::representations::BitsetGenome::popcount)
        .def("encoding_name", &ga::representations::BitsetGenome::encodingName);

    nb::class_<ga::representations::PermutationGenome, ga::IGenome>(m, "PermutationGenome")
        .def(nb::init<>())
        .def(nb::init<std::size_t>(), nb::arg("size"))
        .def(nb::init<std::vector<int>>(), nb::arg("order"))
        .def_rw("order", &ga::representations::PermutationGenome::order)
        .def("size", &ga::representations::PermutationGenome::size)
        .def("is_valid", &ga::representations::PermutationGenome::isValid)
        .def("position_of", &ga::representations::PermutationGenome::positionOf, nb::arg("value"))
        .def_static("random", [](std::size_t n, unsigned seed) {
            std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
            return ga::representations::PermutationGenome::random(n, rng);
        }, nb::arg("size"), nb::arg("seed") = 0u)
        .def("encoding_name", &ga::representations::PermutationGenome::encodingName);

    nb::class_<ga::representations::SetGenome, ga::IGenome>(m, "SetGenome")
        .def(nb::init<>())
        .def(nb::init<std::set<int>>(), nb::arg("values"))
        .def_rw("values", &ga::representations::SetGenome::values)
        .def("encoding_name", &ga::representations::SetGenome::encodingName);

    nb::class_<ga::representations::MapGenome, ga::IGenome>(m, "MapGenome")
        .def(nb::init<>())
        .def(nb::init<std::unordered_map<std::string, double>>(), nb::arg("values"))
        .def_rw("values", &ga::representations::MapGenome::values)
        .def("encoding_name", &ga::representations::MapGenome::encodingName);

    nb::class_<ga::representations::NdArrayGenome, ga::IGenome>(m, "NdArrayGenome")
        .def(nb::init<>())
        .def(nb::init<std::size_t, std::size_t>(), nb::arg("rows"), nb::arg("cols"))
        .def_rw("rows", &ga::representations::NdArrayGenome::rows)
        .def_rw("cols", &ga::representations::NdArrayGenome::cols)
        .def_rw("data", &ga::representations::NdArrayGenome::data)
        .def("get", [](const ga::representations::NdArrayGenome& self, std::size_t r, std::size_t c) {
            return self.at(r, c);
        }, nb::arg("row"), nb::arg("col"))
        .def("set", [](ga::representations::NdArrayGenome& self, std::size_t r, std::size_t c, double value) {
            self.at(r, c) = value;
        }, nb::arg("row"), nb::arg("col"), nb::arg("value"))
        .def("encoding_name", &ga::representations::NdArrayGenome::encodingName);

    nb::enum_<ga::gp::ValueType>(m, "ValueType")
        .value("any", ga::gp::ValueType::Any)
        .value("bool", ga::gp::ValueType::Bool)
        .value("int", ga::gp::ValueType::Int)
        .value("double", ga::gp::ValueType::Double);

    nb::class_<ga::gp::Signature>(m, "Signature")
        .def(nb::init<>())
        .def_rw("return_type", &ga::gp::Signature::returnType)
        .def_rw("arg_types", &ga::gp::Signature::argTypes);

    nb::class_<ga::gp::Primitive>(m, "Primitive")
        .def(nb::init<>())
        .def_rw("name", &ga::gp::Primitive::name)
        .def_rw("signature", &ga::gp::Primitive::signature)
        .def_rw("is_terminal", &ga::gp::Primitive::isTerminal);

    nb::class_<ga::gp::Node>(m, "Node")
        .def(nb::init<>())
        .def(nb::init<std::string, ga::gp::ValueType>(), nb::arg("symbol"), nb::arg("return_type"))
        .def_rw("symbol", &ga::gp::Node::symbol)
        .def_rw("return_type", &ga::gp::Node::returnType)
        .def("size", &ga::gp::Node::size)
        .def("child_count", [](const ga::gp::Node& self) { return self.children.size(); })
        .def("add_child", [](ga::gp::Node& self, const ga::gp::Node& child) {
            self.children.push_back(child.clone());
        });

    nb::class_<ga::representations::TreeGenome, ga::IGenome>(m, "TreeGenome")
        .def(nb::init<>())
        .def("__init__", [](ga::representations::TreeGenome* tg, const ga::gp::Node& root) {
            new (tg) ga::representations::TreeGenome(root.clone());
        }, nb::arg("root"))
        .def("has_root", [](const ga::representations::TreeGenome& self) { return static_cast<bool>(self.root); })
        .def("set_root", [](ga::representations::TreeGenome& self, const ga::gp::Node& root) {
            self.root = root.clone();
        }, nb::arg("root"))
        .def("root", [](const ga::representations::TreeGenome& self) -> nb::object {
            if (!self.root) {
                return nb::none();
            }
            return nb::cast(self.root.get(), nb::rv_policy::reference);
        })
        .def("encoding_name", &ga::representations::TreeGenome::encodingName);

    nb::class_<ga::gp::TreeBuilder>(m, "TreeBuilder")
        .def(nb::init<std::vector<ga::gp::Primitive>>(), nb::arg("primitives"))
        .def("grow", [](const ga::gp::TreeBuilder& self,
                        std::size_t maxDepth,
                        ga::gp::ValueType targetType,
                        bool stronglyTyped,
                        unsigned seed) {
            std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
            return self.grow(maxDepth, targetType, stronglyTyped, rng);
        }, nb::arg("max_depth"), nb::arg("target_type") = ga::gp::ValueType::Any,
           nb::arg("strongly_typed") = false, nb::arg("seed") = 0u);

    nb::class_<ga::gp::ADFPool>(m, "ADFPool")
        .def(nb::init<>())
        .def("put", &ga::gp::ADFPool::put, nb::arg("name"), nb::arg("root"))
        .def("has", &ga::gp::ADFPool::has, nb::arg("name"))
        .def("get", [](const ga::gp::ADFPool& self, const std::string& name) -> const ga::gp::Node& {
            return self.get(name);
        }, nb::arg("name"), nb::rv_policy::reference_internal)
        .def("size", &ga::gp::ADFPool::size);

    // -------------------------------------------------------- GeneticAlgorithm
    nb::class_<ga::GeneticAlgorithm>(m, "GeneticAlgorithm",
        R"(Main Genetic Algorithm class.

        Example
        -------
        >>> import genetic_algorithm_lib as ga
        >>> import math
        >>> cfg = ga.Config()
        >>> cfg.population_size = 60
        >>> cfg.generations = 200
        >>> cfg.dimension = 10
        >>> cfg.bounds = ga.Bounds(-5.12, 5.12)
        >>> engine = ga.GeneticAlgorithm(cfg)
        >>> def sphere(x):
        ...     return 1000.0 / (1.0 + sum(xi**2 for xi in x))
        >>> result = engine.run(sphere)
        >>> print(result.best_fitness)
        )")
        .def(nb::init<const ga::Config&>(), nb::arg("config"))
        .def("run",
             nb::overload_cast<const ga::Fitness&,
                               const std::vector<std::vector<double>>&>(
                 &ga::GeneticAlgorithm::run),
             nb::arg("fitness"), nb::arg("initial_solutions") = std::vector<std::vector<double>>{},
             "Run the GA with the given fitness callable (list[float] -> float). Higher is better.")
        .def("set_mutation_operator",
             [](ga::GeneticAlgorithm& self, nb::object op) {
                 auto cloned = cloneMutation(op);
                 if (cloned) {
                     self.setMutationOperator(std::move(cloned));
                     return;
                 }
                 try {
                     self.setMutationOperator(nb::cast<std::unique_ptr<MutationOperator>>(op));
                 } catch (...) {
                     self.setMutationOperator(std::make_unique<PyMutationOperatorWrapper>(op));
                 }
             },
             nb::arg("op"), "Set a custom mutation operator")
        .def("set_crossover_operator",
             [](ga::GeneticAlgorithm& self, nb::object op) {
                 auto cloned = cloneCrossover(op);
                 if (cloned) {
                     self.setCrossoverOperator(std::move(cloned));
                     return;
                 }
                 try {
                     self.setCrossoverOperator(nb::cast<std::unique_ptr<CrossoverOperator>>(op));
                 } catch (...) {
                     CrossoverOperator* raw = nullptr;
                     if (nb::isinstance<CrossoverOperator>(op)) {
                         raw = nb::cast<CrossoverOperator*>(op);
                     }
                     self.setCrossoverOperator(std::make_unique<PyCrossoverOperatorWrapper>(op, raw));
                 }
             },
             nb::arg("op"), "Set a custom crossover operator")
        .def("config", &ga::GeneticAlgorithm::config,
             nb::rv_policy::reference_internal, "Return the current Config");

    // ------------------------------------------------------------------- NSGA-II
    nb::class_<ga::moea::Nsga2Config>(m, "Nsga2Config", "NSGA-II configuration")
        .def(nb::init<>())
        .def_rw("population_size", &ga::moea::Nsga2Config::populationSize)
        .def_rw("generations", &ga::moea::Nsga2Config::generations)
        .def_rw("seed", &ga::moea::Nsga2Config::seed);

    nb::class_<ga::moea::Nsga2>(m, "Nsga2", "NSGA-II utility methods for objective-space operations")
        .def(nb::init<const ga::moea::Nsga2Config&>(), nb::arg("config") = ga::moea::Nsga2Config{})
        .def("non_dominated_sort_objectives",
             [](const ga::moea::Nsga2& self, const std::vector<std::vector<double>>& objectiveMatrix) {
                 auto population = objectivesToIndividuals(objectiveMatrix);
                 return self.nonDominatedSort(population);
             },
             nb::arg("objectives"),
             "Run non-dominated sorting on objective vectors (minimization)")
        .def("crowding_distance_objectives",
             [](const ga::moea::Nsga2& self,
                const std::vector<std::vector<double>>& objectiveMatrix,
                const std::vector<std::size_t>& front) {
                 auto population = objectivesToIndividuals(objectiveMatrix);
                 return self.crowdingDistance(population, front);
             },
             nb::arg("objectives"),
             nb::arg("front"),
             "Compute crowding distance for one front over objective vectors");

    m.def("nsga2_non_dominated_sort",
          [](const std::vector<std::vector<double>>& objectiveMatrix) {
              ga::moea::Nsga2 nsga2;
              auto population = objectivesToIndividuals(objectiveMatrix);
              return nsga2.nonDominatedSort(population);
          },
          nb::arg("objectives"),
          "Convenience function: non-dominated sorting in objective space");

    m.def("nsga2_crowding_distance",
          [](const std::vector<std::vector<double>>& objectiveMatrix,
             const std::vector<std::size_t>& front) {
              ga::moea::Nsga2 nsga2;
              auto population = objectivesToIndividuals(objectiveMatrix);
              return nsga2.crowdingDistance(population, front);
          },
          nb::arg("objectives"),
          nb::arg("front"),
          "Convenience function: crowding distance in objective space");

    nb::class_<ga::moea::Spea2>(m, "Spea2", "SPEA2 objective-space utilities")
        .def(nb::init<>())
        .def("strength_fitness_objectives",
             [](const ga::moea::Spea2& self, const std::vector<std::vector<double>>& objectiveMatrix) {
                 auto population = objectivesToIndividuals(objectiveMatrix);
                 return self.strengthFitness(population);
             },
             nb::arg("objectives"),
             "Compute SPEA2 strength fitness values in objective space (lower is better)")
        .def("environmental_select_objectives",
             [](const ga::moea::Spea2& self,
                const std::vector<std::vector<double>>& objectiveMatrix,
                std::size_t targetSize) {
                 auto population = objectivesToIndividuals(objectiveMatrix);
                 auto selected = self.environmentalSelect(population, targetSize);
                 return individualsToObjectives(selected);
             },
             nb::arg("objectives"),
             nb::arg("target_size"),
             "Select next generation objective vectors using SPEA2")
        .def("environmental_select_indices",
             [](const ga::moea::Spea2& self,
                const std::vector<std::vector<double>>& objectiveMatrix,
                std::size_t targetSize) {
                 auto population = objectivesToIndividuals(objectiveMatrix, true);
                 auto selected = self.environmentalSelect(population, targetSize);
                 std::vector<std::size_t> indices;
                 indices.reserve(selected.size());
                 for (const auto& ind : selected) {
                     indices.push_back(static_cast<std::size_t>(ind.age));
                 }
                 return indices;
             },
             nb::arg("objectives"),
             nb::arg("target_size"),
             "Select indices into the input objective vectors using SPEA2");

    m.def("spea2_strength_fitness",
          [](const std::vector<std::vector<double>>& objectiveMatrix) {
              ga::moea::Spea2 spea2;
              auto population = objectivesToIndividuals(objectiveMatrix);
              return spea2.strengthFitness(population);
          },
          nb::arg("objectives"),
          "Convenience function: SPEA2 strength fitness in objective space");

    m.def("spea2_environmental_select_indices",
          [](const std::vector<std::vector<double>>& objectiveMatrix, std::size_t targetSize) {
              ga::moea::Spea2 spea2;
              auto population = objectivesToIndividuals(objectiveMatrix, true);
              auto selected = spea2.environmentalSelect(population, targetSize);
              std::vector<std::size_t> indices;
              indices.reserve(selected.size());
              for (const auto& ind : selected) {
                  indices.push_back(static_cast<std::size_t>(ind.age));
              }
              return indices;
          },
          nb::arg("objectives"),
          nb::arg("target_size"),
          "Convenience function: SPEA2 environmental selection indices");

    // ------------------------------------------------------------------- NSGA-III
    nb::class_<ga::moea::Nsga3>(m, "Nsga3", "NSGA-III utility methods for objective-space operations")
        .def(nb::init<ga::moea::Nsga2Config>(), nb::arg("config") = ga::moea::Nsga2Config{})
        .def_static("generate_reference_points",
                    &ga::moea::Nsga3::generateDasDennisReferencePoints,
                    nb::arg("objective_count"),
                    nb::arg("divisions"),
                    "Generate Das-Dennis reference points")
        .def("non_dominated_sort_objectives",
             [](const ga::moea::Nsga3& self,
                const std::vector<std::vector<double>>& objectiveMatrix) {
                 auto population = objectivesToIndividuals(objectiveMatrix);
                 return self.nonDominatedSort(population);
             },
             nb::arg("objectives"),
             "Run non-dominated sorting on objective vectors (minimization)")
        .def("environmental_select_objectives",
             [](const ga::moea::Nsga3& self,
                const std::vector<std::vector<double>>& objectiveMatrix,
                std::size_t targetSize,
                const std::vector<std::vector<double>>& referencePoints) {
                 auto population = objectivesToIndividuals(objectiveMatrix);
                 auto selected = self.environmentalSelect(population, targetSize, referencePoints);
                 return individualsToObjectives(selected);
             },
             nb::arg("objectives"),
             nb::arg("target_size"),
             nb::arg("reference_points"),
             "Select next generation objective vectors using NSGA-III niching")
        .def("environmental_select_indices",
             [](const ga::moea::Nsga3& self,
                const std::vector<std::vector<double>>& objectiveMatrix,
                std::size_t targetSize,
                const std::vector<std::vector<double>>& referencePoints) {
                 auto population = objectivesToIndividuals(objectiveMatrix, true);
                 auto selected = self.environmentalSelect(population, targetSize, referencePoints);
                 std::vector<std::size_t> indices;
                 indices.reserve(selected.size());
                 for (const auto& ind : selected) {
                     indices.push_back(static_cast<std::size_t>(ind.age));
                 }
                 return indices;
             },
             nb::arg("objectives"),
             nb::arg("target_size"),
             nb::arg("reference_points"),
             "Select indices into the input objective vectors using NSGA-III niching");

    m.def("nsga3_reference_points",
          &ga::moea::Nsga3::generateDasDennisReferencePoints,
          nb::arg("objective_count"),
          nb::arg("divisions"),
          "Convenience function: generate NSGA-III Das-Dennis reference points");

    m.def("nsga3_environmental_select_indices",
          [](const std::vector<std::vector<double>>& objectiveMatrix,
             std::size_t targetSize,
             const std::vector<std::vector<double>>& referencePoints) {
              ga::moea::Nsga3 nsga3;
              auto population = objectivesToIndividuals(objectiveMatrix, true);
              auto selected = nsga3.environmentalSelect(population, targetSize, referencePoints);
              std::vector<std::size_t> indices;
              indices.reserve(selected.size());
              for (const auto& ind : selected) {
                  indices.push_back(static_cast<std::size_t>(ind.age));
              }
              return indices;
          },
          nb::arg("objectives"),
          nb::arg("target_size"),
          nb::arg("reference_points"),
          "Convenience function: NSGA-III environmental selection over objective vectors");

    // -------------------------------------------------------------- Checkpoint API
    nb::class_<ga::checkpoint::CheckpointState>(m, "CheckpointState", "Checkpoint serialization state")
        .def(nb::init<>())
        .def_rw("config", &ga::checkpoint::CheckpointState::config)
        .def_rw("result", &ga::checkpoint::CheckpointState::result)
        .def_rw("generation", &ga::checkpoint::CheckpointState::generation)
        .def_rw("rng_state", &ga::checkpoint::CheckpointState::rngState);

    m.def("checkpoint_save_json",
          [](const std::string& path, const ga::checkpoint::CheckpointState& state) {
              ga::checkpoint::CheckpointManager::saveJson(path, state);
          },
          nb::arg("path"),
          nb::arg("state"),
          "Save checkpoint state as JSON");

    m.def("checkpoint_load_json",
          [](const std::string& path) {
              return ga::checkpoint::CheckpointManager::loadJson(path);
          },
          nb::arg("path"),
          "Load checkpoint state from JSON");

    m.def("checkpoint_save_binary",
          [](const std::string& path, const ga::checkpoint::CheckpointState& state) {
              ga::checkpoint::CheckpointManager::saveBinary(path, state);
          },
          nb::arg("path"),
          nb::arg("state"),
          "Save checkpoint state as binary");

    m.def("checkpoint_load_binary",
          [](const std::string& path) {
              return ga::checkpoint::CheckpointManager::loadBinary(path);
          },
          nb::arg("path"),
          "Load checkpoint state from binary");

    // -------------------------------------------------------------------- Optimizer API
    nb::class_<ga::api::Optimizer::MultiObjectiveResult>(m, "MultiObjectiveResult",
                                                          "Multi-objective optimizer result")
        .def(nb::init<>())
        .def_rw("pareto_genes", &ga::api::Optimizer::MultiObjectiveResult::paretoGenes)
        .def_rw("pareto_objectives", &ga::api::Optimizer::MultiObjectiveResult::paretoObjectives);

    nb::class_<ga::api::Optimizer>(m, "Optimizer", "High-level optimizer facade")
        .def(nb::init<>())
        .def("with_config", &ga::api::Optimizer::withConfig, nb::arg("config"),
             nb::rv_policy::reference_internal)
        .def("with_threads", &ga::api::Optimizer::withThreads, nb::arg("threads"),
             nb::rv_policy::reference_internal)
        .def("with_seed", &ga::api::Optimizer::withSeed, nb::arg("seed"),
             nb::rv_policy::reference_internal)
        .def("optimize", &ga::api::Optimizer::optimize, nb::arg("objective"))
        .def("optimize_multi_objective_nsga2",
             [](const ga::api::Optimizer& self, const nb::iterable& objectiveCallables,
                std::size_t populationSize, std::size_t generations) {
                 auto objectives = pyObjectivesToCpp(objectiveCallables);
                 return self.optimizeMultiObjectiveNsga2(objectives, populationSize, generations);
             },
             nb::arg("objectives"), nb::arg("population_size") = 80, nb::arg("generations") = 80)
        .def("optimize_multi_objective_nsga3",
             [](const ga::api::Optimizer& self, const nb::iterable& objectiveCallables,
                std::size_t populationSize, std::size_t generations, std::size_t referenceDivisions) {
                 auto objectives = pyObjectivesToCpp(objectiveCallables);
                 return self.optimizeMultiObjectiveNsga3(
                     objectives, populationSize, generations, referenceDivisions);
             },
             nb::arg("objectives"), nb::arg("population_size") = 80, nb::arg("generations") = 80,
             nb::arg("reference_divisions") = 8);

    nb::class_<ga::api::OptimizerBuilder>(m, "OptimizerBuilder", "Fluent optimizer builder")
        .def(nb::init<>())
        .def("dimension", &ga::api::OptimizerBuilder::dimension, nb::arg("dimension"),
             nb::rv_policy::reference_internal)
        .def("bounds", &ga::api::OptimizerBuilder::bounds, nb::arg("lower"), nb::arg("upper"),
             nb::rv_policy::reference_internal)
        .def("population_size", &ga::api::OptimizerBuilder::populationSize, nb::arg("population_size"),
             nb::rv_policy::reference_internal)
        .def("generations", &ga::api::OptimizerBuilder::generations, nb::arg("generations"),
             nb::rv_policy::reference_internal)
        .def("seed", &ga::api::OptimizerBuilder::seed, nb::arg("seed"),
             nb::rv_policy::reference_internal)
        .def("crossover_rate", &ga::api::OptimizerBuilder::crossoverRate, nb::arg("crossover_rate"),
             nb::rv_policy::reference_internal)
        .def("mutation_rate", &ga::api::OptimizerBuilder::mutationRate, nb::arg("mutation_rate"),
             nb::rv_policy::reference_internal)
        .def("elite_ratio", &ga::api::OptimizerBuilder::eliteRatio, nb::arg("elite_ratio"),
             nb::rv_policy::reference_internal)
        .def("threads", &ga::api::OptimizerBuilder::threads, nb::arg("threads"),
             nb::rv_policy::reference_internal)
        .def("build", &ga::api::OptimizerBuilder::build);

    // ------------------------------------------------------------ ES / CMA-ES / MO-CMA-ES
    nb::class_<ga::es::EvolutionStrategyConfig>(m, "EvolutionStrategyConfig")
        .def(nb::init<>())
        .def_rw("mu", &ga::es::EvolutionStrategyConfig::mu)
        .def_rw("lambda_", &ga::es::EvolutionStrategyConfig::lambda)
        .def_rw("generations", &ga::es::EvolutionStrategyConfig::generations)
        .def_rw("dimension", &ga::es::EvolutionStrategyConfig::dimension)
        .def_rw("sigma", &ga::es::EvolutionStrategyConfig::sigma)
        .def_rw("lower", &ga::es::EvolutionStrategyConfig::lower)
        .def_rw("upper", &ga::es::EvolutionStrategyConfig::upper)
        .def_rw("plus_strategy", &ga::es::EvolutionStrategyConfig::plusStrategy)
        .def_rw("seed", &ga::es::EvolutionStrategyConfig::seed);

    nb::class_<ga::es::EvolutionStrategyResult>(m, "EvolutionStrategyResult")
        .def(nb::init<>())
        .def_rw("best", &ga::es::EvolutionStrategyResult::best)
        .def_rw("best_fitness", &ga::es::EvolutionStrategyResult::bestFitness)
        .def_rw("best_history", &ga::es::EvolutionStrategyResult::bestHistory);

    nb::class_<ga::es::EvolutionStrategy>(m, "EvolutionStrategy")
        .def(nb::init<ga::es::EvolutionStrategyConfig>(), nb::arg("config"))
        .def("run", &ga::es::EvolutionStrategy::run, nb::arg("fitness"));

    nb::class_<ga::es::CmaEsConfig>(m, "CmaEsConfig")
        .def(nb::init<>())
        .def_rw("population_size", &ga::es::CmaEsConfig::populationSize)
        .def_rw("generations", &ga::es::CmaEsConfig::generations)
        .def_rw("dimension", &ga::es::CmaEsConfig::dimension)
        .def_rw("lower", &ga::es::CmaEsConfig::lower)
        .def_rw("upper", &ga::es::CmaEsConfig::upper)
        .def_rw("sigma", &ga::es::CmaEsConfig::sigma)
        .def_rw("seed", &ga::es::CmaEsConfig::seed);

    nb::class_<ga::es::CmaEsResult>(m, "CmaEsResult")
        .def(nb::init<>())
        .def_rw("best", &ga::es::CmaEsResult::best)
        .def_rw("best_fitness", &ga::es::CmaEsResult::bestFitness)
        .def_rw("history", &ga::es::CmaEsResult::history);

    nb::class_<ga::es::DiagonalCmaEs>(m, "DiagonalCmaEs")
        .def(nb::init<ga::es::CmaEsConfig>(), nb::arg("config"))
        .def("run", &ga::es::DiagonalCmaEs::run, nb::arg("fitness"));

    nb::class_<ga::moea::MoCmaEsConfig>(m, "MoCmaEsConfig")
        .def(nb::init<>())
        .def_rw("cma", &ga::moea::MoCmaEsConfig::cma)
        .def_rw("weights", &ga::moea::MoCmaEsConfig::weights);

    nb::class_<ga::moea::MoCmaEsResult>(m, "MoCmaEsResult")
        .def(nb::init<>())
        .def_rw("best", &ga::moea::MoCmaEsResult::best)
        .def_rw("objectives", &ga::moea::MoCmaEsResult::objectives)
        .def_rw("weighted_fitness", &ga::moea::MoCmaEsResult::weightedFitness);

    nb::class_<ga::moea::MoCmaEs>(m, "MoCmaEs")
        .def(nb::init<ga::moea::MoCmaEsConfig>(), nb::arg("config"))
        .def("run", &ga::moea::MoCmaEs::run, nb::arg("objective"));

    // ----------------------------------------------------- Constraints and adaptation
    nb::class_<ga::constraints::ConstraintSet>(m, "ConstraintSet")
        .def(nb::init<>())
        .def("add_hard_constraint", [](ga::constraints::ConstraintSet& self, nb::callable fn) {
            self.hard.emplace_back([fn](const std::vector<double>& genes) {
                nb::gil_scoped_acquire acquire;
                return nb::cast<bool>(fn(genes));
            });
        }, nb::arg("constraint"))
        .def("add_soft_penalty", [](ga::constraints::ConstraintSet& self, nb::callable fn) {
            self.soft.emplace_back([fn](const std::vector<double>& genes) {
                nb::gil_scoped_acquire acquire;
                return nb::cast<double>(fn(genes));
            });
        }, nb::arg("penalty"))
        .def("add_repair", [](ga::constraints::ConstraintSet& self, nb::callable fn) {
            self.repairs.emplace_back([fn](std::vector<double>& genes) {
                nb::gil_scoped_acquire acquire;
                nb::object out = fn(genes);
                if (!out.is_none()) {
                    genes = nb::cast<std::vector<double>>(out);
                }
            });
        }, nb::arg("repair"))
        .def("clear", [](ga::constraints::ConstraintSet& self) {
            self.hard.clear();
            self.soft.clear();
            self.repairs.clear();
        });

    m.def("is_feasible", &ga::constraints::isFeasible, nb::arg("genes"), nb::arg("constraint_set"));
    m.def("total_penalty", &ga::constraints::totalPenalty, nb::arg("genes"), nb::arg("constraint_set"));
    m.def("apply_repairs", [](std::vector<double> genes, const ga::constraints::ConstraintSet& set) {
        ga::constraints::applyRepairs(genes, set);
        return genes;
    }, nb::arg("genes"), nb::arg("constraint_set"));
    m.def("penalized_fitness", &ga::constraints::penalizedFitness,
          nb::arg("base_fitness"), nb::arg("genes"), nb::arg("constraint_set"),
          nb::arg("infeasible_penalty") = 1e6);

    nb::class_<ga::constraints::DebProfile>(m, "DebProfile")
        .def(nb::init<>())
        .def_rw("fitness", &ga::constraints::DebProfile::fitness)
        .def_rw("violation", &ga::constraints::DebProfile::violation)
        .def_rw("is_feasible", &ga::constraints::DebProfile::isFeasible);

    m.def("deb_is_better", &ga::constraints::debIsBetter,
          nb::arg("a"), nb::arg("b"),
          "Compares two candidate profiles using Deb's feasibility rules (Deb 2000)");

    nb::class_<ga::constraints::AdaptivePenaltyConfig>(m, "AdaptivePenaltyConfig")
        .def(nb::init<>())
        .def_rw("base_coefficient", &ga::constraints::AdaptivePenaltyConfig::baseCoefficient)
        .def_rw("alpha", &ga::constraints::AdaptivePenaltyConfig::alpha)
        .def_rw("beta", &ga::constraints::AdaptivePenaltyConfig::beta)
        .def_rw("min_penalty", &ga::constraints::AdaptivePenaltyConfig::minPenalty)
        .def_rw("max_penalty", &ga::constraints::AdaptivePenaltyConfig::maxPenalty);

    nb::class_<ga::constraints::AdaptivePenaltyHandler>(m, "AdaptivePenaltyHandler")
        .def(nb::init<ga::constraints::AdaptivePenaltyConfig>(),
             nb::arg("config") = ga::constraints::AdaptivePenaltyConfig{})
        .def("penalty_factor", &ga::constraints::AdaptivePenaltyHandler::penaltyFactor, nb::arg("iteration"))
        .def("penalize", &ga::constraints::AdaptivePenaltyHandler::penalize,
             nb::arg("raw_fitness"), nb::arg("violation"), nb::arg("iteration"));

    nb::class_<ga::constraints::ConstrainedOptimizerConfig>(m, "ConstrainedOptimizerConfig")
        .def(nb::init<>())
        .def_rw("search", &ga::constraints::ConstrainedOptimizerConfig::search)
        .def_rw("use_deb_feasibility", &ga::constraints::ConstrainedOptimizerConfig::useDebFeasibility)
        .def_rw("use_adaptive_penalty", &ga::constraints::ConstrainedOptimizerConfig::useAdaptivePenalty)
        .def_rw("penalty_config", &ga::constraints::ConstrainedOptimizerConfig::penaltyConfig)
        .def_rw("crossover_rate", &ga::constraints::ConstrainedOptimizerConfig::crossoverRate)
        .def_rw("mutation_rate", &ga::constraints::ConstrainedOptimizerConfig::mutationRate)
        .def_rw("tournament_size", &ga::constraints::ConstrainedOptimizerConfig::tournamentSize);

    nb::class_<ga::constraints::ConstrainedOptimizer,
               ga::metaheuristics::IContinuousOptimizer>(m, "ConstrainedOptimizer")
        .def(nb::init<ga::constraints::ConstrainedOptimizerConfig, ga::constraints::ConstraintSet>(),
             nb::arg("config") = ga::constraints::ConstrainedOptimizerConfig{},
             nb::arg("constraints") = ga::constraints::ConstraintSet{})
        .def("name", &ga::constraints::ConstrainedOptimizer::name)
        .def("optimize",
             [](ga::constraints::ConstrainedOptimizer& self,
                nb::callable fitness,
                const ga::metaheuristics::SeedPopulation& seeds) {
                 return optimizeFromPython(self, std::move(fitness), seeds);
             },
             nb::arg("fitness"),
             nb::arg("seeds") = ga::metaheuristics::SeedPopulation{});

    m.def("run_constrained_optimization", [](nb::callable fitness,
                                            const ga::constraints::ConstraintSet& constraints,
                                            std::size_t dimension,
                                            std::size_t iterations,
                                            ga::Bounds bounds,
                                            std::size_t population_size,
                                            unsigned seed) {
        ga::Fitness wrapped = wrapPythonFitness(std::move(fitness));
        nb::gil_scoped_release release;
        return ga::api::runConstrainedOptimization(wrapped, constraints, dimension, iterations, bounds, population_size, seed);
    }, nb::arg("fitness"),
       nb::arg("constraints") = ga::constraints::ConstraintSet{},
       nb::arg("dimension") = 10,
       nb::arg("iterations") = 100,
       nb::arg("bounds") = ga::Bounds{-5.12, 5.12},
       nb::arg("population_size") = 50,
       nb::arg("seed") = 0);

    nb::class_<ga::adaptive::AdaptiveRates>(m, "AdaptiveRates")
        .def(nb::init<>())
        .def_rw("mutation_rate", &ga::adaptive::AdaptiveRates::mutationRate)
        .def_rw("crossover_rate", &ga::adaptive::AdaptiveRates::crossoverRate);

    nb::class_<ga::adaptive::AdaptiveRateController>(m, "AdaptiveRateController")
        .def(nb::init<double, double, double, double>(),
             nb::arg("min_mutation") = 0.001, nb::arg("max_mutation") = 0.6,
             nb::arg("min_crossover") = 0.4, nb::arg("max_crossover") = 0.95)
        .def("update", &ga::adaptive::AdaptiveRateController::update, nb::arg("current"),
             nb::arg("diversity"), nb::arg("best_improvement"));

    // ---------------------------------------------------------- Hybrid / Coevolution
    nb::class_<ga::hybrid::HybridOptimizer>(m, "HybridOptimizer")
        .def(nb::init<ga::Config>(), nb::arg("config"))
        .def("run",
             [](const ga::hybrid::HybridOptimizer& self,
                const ga::Fitness& fitness,
                nb::object localSearch,
                std::size_t localSearchRestarts) {
                 ga::hybrid::HybridOptimizer::LocalSearch ls;
                 if (!localSearch.is_none()) {
                     ls = [localSearch](std::vector<double>& genes) {
                         nb::gil_scoped_acquire acquire;
                         nb::object out = localSearch(genes);
                         if (!out.is_none()) {
                             genes = nb::cast<std::vector<double>>(out);
                         }
                     };
                 }
                 return self.run(fitness, ls, localSearchRestarts);
             },
             nb::arg("fitness"),
             nb::arg("local_search") = nb::none(),
             nb::arg("local_search_restarts") = 5);

    nb::class_<ga::coevolution::CoevolutionConfig>(m, "CoevolutionConfig")
        .def(nb::init<>())
        .def_rw("generations", &ga::coevolution::CoevolutionConfig::generations)
        .def_rw("seed", &ga::coevolution::CoevolutionConfig::seed);

    nb::class_<ga::coevolution::CoevolutionEngine>(m, "CoevolutionEngine")
        .def(nb::init<ga::coevolution::CoevolutionConfig>(), nb::arg("config"))
        .def("run",
             [](const ga::coevolution::CoevolutionEngine& self,
                ga::coevolution::CoevolutionEngine::Populations populations,
                nb::object evaluate,
                nb::object reproduce) {
                 ga::coevolution::CoevolutionEngine::EvaluateFn evalFn;
                 if (!evaluate.is_none()) {
                     evalFn = [evaluate](ga::coevolution::CoevolutionEngine::Populations& pops) {
                         nb::gil_scoped_acquire acquire;
                         evaluate(pops);
                     };
                 }
                 ga::coevolution::CoevolutionEngine::ReproduceFn repFn;
                 if (!reproduce.is_none()) {
                     repFn = [reproduce](ga::coevolution::CoevolutionEngine::Populations& pops, std::mt19937&) {
                         nb::gil_scoped_acquire acquire;
                         reproduce(pops);
                     };
                 }
                 return self.run(std::move(populations), evalFn, repFn);
             },
             nb::arg("populations"),
             nb::arg("evaluate") = nb::none(),
             nb::arg("reproduce") = nb::none());

    // ------------------------------------------------------ Tracking / visualization
    nb::class_<ga::tracking::ExperimentTracker>(m, "ExperimentTracker")
        .def(nb::init<std::string>(), nb::arg("run_id"))
        .def("write_config", &ga::tracking::ExperimentTracker::writeConfig,
             nb::arg("config"), nb::arg("path"))
        .def("write_history_csv", &ga::tracking::ExperimentTracker::writeHistoryCSV,
             nb::arg("result"), nb::arg("path"))
        .def("write_best_solution_csv", &ga::tracking::ExperimentTracker::writeBestSolutionCSV,
             nb::arg("result"), nb::arg("path"));

    m.def("export_fitness_curve_csv", &ga::visualization::exportFitnessCurveCSV,
          nb::arg("best"), nb::arg("avg"), nb::arg("path"));
    m.def("export_pareto_front_csv", &ga::visualization::exportParetoFrontCSV,
          nb::arg("objectives"), nb::arg("path"));
    m.def("export_diversity_csv", &ga::visualization::exportDiversityCSV,
          nb::arg("diversity"), nb::arg("path"));

    // ------------------------------------------------------- Evaluation helpers
    nb::class_<DoubleBatchEvaluator>(m, "ParallelEvaluator",
                                     "Threaded batch evaluator over vector<double> candidates")
        .def("__init__", [](DoubleBatchEvaluator* be, nb::callable fitness, std::size_t threads) {
                 std::function<double(const std::vector<double>&)> wrapped =
                     [fitness](const std::vector<double>& genes) {
                         nb::gil_scoped_acquire acquire;
                         return nb::cast<double>(fitness(genes));
                     };
                 new (be) DoubleBatchEvaluator(std::move(wrapped), threads);
             },
             nb::arg("fitness"),
             nb::arg("threads") = std::thread::hardware_concurrency())
        .def("evaluate", &DoubleBatchEvaluator::evaluate, nb::arg("batch"),
             nb::call_guard<nb::gil_scoped_release>(),
             "Evaluate a batch of candidate vectors in parallel");

    nb::class_<ga::evaluation::LocalDistributedExecutor>(m, "LocalDistributedExecutor",
                                                          "Local threaded distributed executor")
        .def("__init__", [](ga::evaluation::LocalDistributedExecutor* le, nb::callable evaluator, std::size_t workers) {
                 ga::evaluation::LocalDistributedExecutor::EvaluateFn wrapped =
                     [evaluator](const std::vector<double>& genes) {
                         nb::gil_scoped_acquire acquire;
                         return nb::cast<double>(evaluator(genes));
                     };
                 new (le) ga::evaluation::LocalDistributedExecutor(
                     std::move(wrapped), workers);
             },
             nb::arg("evaluator"),
             nb::arg("workers") = std::thread::hardware_concurrency())
        .def("execute", &ga::evaluation::LocalDistributedExecutor::execute, nb::arg("batch"),
             nb::call_guard<nb::gil_scoped_release>(),
             "Execute a batch of candidate vectors and return fitness values");

    // ------------------------------------------------------- Benchmark suite
    nb::class_<BenchmarkConfig>(m, "BenchmarkConfig", "Benchmark configuration")
        .def(nb::init<>())
        .def_rw("warmup_iterations", &BenchmarkConfig::warmupIterations)
        .def_rw("benchmark_iterations", &BenchmarkConfig::benchmarkIterations)
        .def_rw("verbose", &BenchmarkConfig::verbose)
        .def_rw("csv_output", &BenchmarkConfig::csvOutput)
        .def_rw("output_file", &BenchmarkConfig::outputFile);

    nb::class_<BenchmarkResult>(m, "BenchmarkResult", "Scalability benchmark aggregate result")
        .def_ro("name", &BenchmarkResult::name)
        .def_ro("category", &BenchmarkResult::category)
        .def_ro("avg_execution_time", &BenchmarkResult::avgExecutionTime)
        .def_ro("min_execution_time", &BenchmarkResult::minExecutionTime)
        .def_ro("max_execution_time", &BenchmarkResult::maxExecutionTime)
        .def_ro("iterations", &BenchmarkResult::iterations)
        .def_ro("throughput", &BenchmarkResult::throughput)
        .def_ro("standard_deviation", &BenchmarkResult::standardDeviation)
        .def_ro("success", &BenchmarkResult::success)
        .def_ro("error_message", &BenchmarkResult::errorMessage);

    nb::class_<OperatorBenchmark>(m, "OperatorBenchmark", "Operator-level benchmark result")
        .def_ro("operator_name", &OperatorBenchmark::operatorName)
        .def_ro("operator_type", &OperatorBenchmark::operatorType)
        .def_ro("avg_time", &OperatorBenchmark::avgTime)
        .def_ro("operations_per_second", &OperatorBenchmark::operationsPerSecond)
        .def_ro("iterations", &OperatorBenchmark::iterations)
        .def_ro("representation", &OperatorBenchmark::representation);

    nb::class_<FunctionBenchmark>(m, "FunctionBenchmark", "Function optimization benchmark result")
        .def_ro("function_name", &FunctionBenchmark::functionName)
        .def_ro("best_fitness", &FunctionBenchmark::bestFitness)
        .def_ro("avg_fitness", &FunctionBenchmark::avgFitness)
        .def_ro("generations_to_converge", &FunctionBenchmark::generationsToConverge)
        .def_ro("total_execution_time", &FunctionBenchmark::totalExecutionTime)
        .def_ro("best_solution", &FunctionBenchmark::bestSolution)
        .def_ro("convergence_history", &FunctionBenchmark::convergenceHistory);

    nb::class_<GABenchmark>(m, "GABenchmark", "Benchmark suite runner")
        .def(nb::init<const BenchmarkConfig&>(), nb::arg("config") = BenchmarkConfig{})
        .def("run_all_benchmarks", &GABenchmark::runAllBenchmarks)
        .def("run_operator_benchmarks", &GABenchmark::runOperatorBenchmarks)
        .def("run_function_benchmarks", &GABenchmark::runFunctionBenchmarks)
        .def("run_scalability_benchmarks", &GABenchmark::runScalabilityBenchmarks)
        .def("generate_report", &GABenchmark::generateReport)
        .def("export_to_csv", &GABenchmark::exportToCSV, nb::arg("filename"))
        .def("operator_results", [](const GABenchmark& self) { return self.operatorResults(); })
        .def("function_results", [](const GABenchmark& self) { return self.functionResults(); })
        .def("scalability_results", [](const GABenchmark& self) { return self.scalabilityResults(); });

    // ------------------------------------------------------- Operator classes
    nb::class_<CrossoverOperator>(m, "CrossoverOperator", "Base class for crossover operators")
        .def_prop_ro("name", &CrossoverOperator::getName)
        .def_prop_ro("operation_count", &CrossoverOperator::getOperationCount)
        .def_prop_ro("error_count", &CrossoverOperator::getErrorCount)
        .def_prop_ro("error_rate", &CrossoverOperator::getErrorRate)
        .def("reset_statistics", &CrossoverOperator::resetStatistics);

    nb::class_<ga::fuzzy::FuzzyAdaptiveCrossoverConfig>(m, "FuzzyAdaptiveCrossoverConfig")
        .def(nb::init<>())
        .def_rw("search_diameter", &ga::fuzzy::FuzzyAdaptiveCrossoverConfig::searchDiameter)
        .def_rw("default_alpha", &ga::fuzzy::FuzzyAdaptiveCrossoverConfig::defaultAlpha)
        .def_rw("default_eta_c", &ga::fuzzy::FuzzyAdaptiveCrossoverConfig::defaultEtaC)
        .def_rw("seed", &ga::fuzzy::FuzzyAdaptiveCrossoverConfig::seed);

    nb::class_<ga::fuzzy::FuzzyAdaptiveCrossover, CrossoverOperator>(m, "FuzzyAdaptiveCrossover")
        .def(nb::init<ga::fuzzy::FuzzyAdaptiveCrossoverConfig>(), nb::arg("config") = ga::fuzzy::FuzzyAdaptiveCrossoverConfig{})
        .def("update_context", &ga::fuzzy::FuzzyAdaptiveCrossover::updateContext, nb::arg("diversity"), nb::arg("relative_gen"))
        .def_prop_ro("config", &ga::fuzzy::FuzzyAdaptiveCrossover::config);

    nb::class_<OnePointCrossover, CrossoverOperator>(m, "OnePointCrossover")
        .def("__init__", [](OnePointCrossover* self, unsigned seed) {
        if (seed == 0) new (self) OnePointCrossover{};
        else new (self) OnePointCrossover{seed};
    }, nb::arg("seed") = 0u)
        .def("crossover_real",
             [](OnePointCrossover& self, const std::vector<double>& parent1, const std::vector<double>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"))
        .def("crossover_bits",
             [](OnePointCrossover& self, const std::vector<bool>& parent1, const std::vector<bool>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"))
        .def("crossover_int",
             [](OnePointCrossover& self, const std::vector<int>& parent1, const std::vector<int>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"));

    nb::class_<TwoPointCrossover, CrossoverOperator>(m, "TwoPointCrossover")
        .def("__init__", [](TwoPointCrossover* self, unsigned seed) {
        if (seed == 0) new (self) TwoPointCrossover{};
        else new (self) TwoPointCrossover{seed};
    }, nb::arg("seed") = 0u)
        .def("crossover_real",
             [](TwoPointCrossover& self, const std::vector<double>& parent1, const std::vector<double>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"))
        .def("crossover_bits",
             [](TwoPointCrossover& self, const std::vector<bool>& parent1, const std::vector<bool>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"))
        .def("crossover_int",
             [](TwoPointCrossover& self, const std::vector<int>& parent1, const std::vector<int>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"));

    nb::class_<UniformCrossover, CrossoverOperator>(m, "UniformCrossover")
        .def("__init__", [](UniformCrossover* self, double probability, unsigned seed) {
        if (seed == 0) new (self) UniformCrossover{probability};
        else new (self) UniformCrossover{probability, seed};
    }, nb::arg("probability") = 0.5, nb::arg("seed") = 0u)
        .def("set_probability", &UniformCrossover::setProbability, nb::arg("probability"))
        .def("get_probability", &UniformCrossover::getProbability)
        .def("crossover_real",
             [](UniformCrossover& self, const std::vector<double>& parent1, const std::vector<double>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"))
        .def("crossover_bits",
             [](UniformCrossover& self, const std::vector<bool>& parent1, const std::vector<bool>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"))
        .def("crossover_int",
             [](UniformCrossover& self, const std::vector<int>& parent1, const std::vector<int>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"));

    nb::class_<MultiPointCrossover, CrossoverOperator>(m, "MultiPointCrossover")
        .def("__init__", [](MultiPointCrossover* self, int points, unsigned seed) {
        if (seed == 0) new (self) MultiPointCrossover{points};
        else new (self) MultiPointCrossover{points, seed};
    }, nb::arg("points") = 3, nb::arg("seed") = 0u)
        .def("set_num_points", &MultiPointCrossover::setNumPoints, nb::arg("points"))
        .def("get_num_points", &MultiPointCrossover::getNumPoints)
        .def("crossover_real",
             [](MultiPointCrossover& self, const std::vector<double>& parent1, const std::vector<double>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"))
        .def("crossover_bits",
             [](MultiPointCrossover& self, const std::vector<bool>& parent1, const std::vector<bool>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"))
        .def("crossover_int",
             [](MultiPointCrossover& self, const std::vector<int>& parent1, const std::vector<int>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"));

    nb::class_<BlendCrossover, CrossoverOperator>(m, "BlendCrossover")
        .def("__init__", [](BlendCrossover* self, double alpha, unsigned seed) {
        if (seed == 0) new (self) BlendCrossover{alpha};
        else new (self) BlendCrossover{alpha, seed};
    }, nb::arg("alpha") = 0.5, nb::arg("seed") = 0u)
        .def("set_alpha", &BlendCrossover::setAlpha, nb::arg("alpha"))
        .def("get_alpha", &BlendCrossover::getAlpha)
        .def("crossover_real",
             [](BlendCrossover& self, const std::vector<double>& parent1, const std::vector<double>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"));

    nb::class_<SimulatedBinaryCrossover, CrossoverOperator>(m, "SimulatedBinaryCrossover")
        .def("__init__", [](SimulatedBinaryCrossover* self, double eta, unsigned seed) {
        if (seed == 0) new (self) SimulatedBinaryCrossover{eta};
        else new (self) SimulatedBinaryCrossover{eta, seed};
    }, nb::arg("eta") = 2.0, nb::arg("seed") = 0u)
        .def("set_eta", &SimulatedBinaryCrossover::setEta, nb::arg("eta"))
        .def("get_eta", &SimulatedBinaryCrossover::getEta)
        .def("crossover_real",
             [](SimulatedBinaryCrossover& self, const std::vector<double>& parent1, const std::vector<double>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"));

    nb::class_<LineRecombination, CrossoverOperator>(m, "LineRecombination")
        .def("__init__", [](LineRecombination* self, double extension_factor, unsigned seed) {
        if (seed == 0) new (self) LineRecombination{extension_factor};
        else new (self) LineRecombination{extension_factor, seed};
    }, nb::arg("extension_factor") = 0.1, nb::arg("seed") = 0u)
        .def("crossover_real",
             [](LineRecombination& self, const std::vector<double>& parent1, const std::vector<double>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"));

    nb::class_<IntermediateRecombination, CrossoverOperator>(m, "IntermediateRecombination")
        .def("__init__", [](IntermediateRecombination* self, double alpha, unsigned seed) {
        if (seed == 0) new (self) IntermediateRecombination{alpha};
        else new (self) IntermediateRecombination{alpha, seed};
    }, nb::arg("alpha") = 0.5, nb::arg("seed") = 0u)
        .def("crossover_real",
             [](IntermediateRecombination& self, const std::vector<double>& parent1, const std::vector<double>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"))
        .def("single_arithmetic_recombination", &IntermediateRecombination::singleArithmeticRecombination,
             nb::arg("parent1"), nb::arg("parent2"))
        .def("whole_arithmetic_recombination", &IntermediateRecombination::wholeArithmeticRecombination,
             nb::arg("parent1"), nb::arg("parent2"));

    nb::class_<DifferentialEvolutionCrossover, CrossoverOperator>(m, "DifferentialEvolutionCrossover")
        .def("__init__", [](DifferentialEvolutionCrossover* self, double crossover_rate, unsigned seed) {
        if (seed == 0) new (self) DifferentialEvolutionCrossover{crossover_rate};
        else new (self) DifferentialEvolutionCrossover{crossover_rate, seed};
    }, nb::arg("crossover_rate") = 0.5, nb::arg("seed") = 0u)
        .def("perform_crossover", &DifferentialEvolutionCrossover::performCrossover,
             nb::arg("target"), nb::arg("mutant"));

    nb::class_<UniformKVectorCrossover, CrossoverOperator>(m, "UniformKVectorCrossover")
        .def("__init__", [](UniformKVectorCrossover* self, double swap_probability, unsigned seed) {
        if (seed == 0) new (self) UniformKVectorCrossover{swap_probability};
        else new (self) UniformKVectorCrossover{swap_probability, seed};
    }, nb::arg("swap_probability") = 0.1, nb::arg("seed") = 0u)
        .def("crossover_real",
             [](UniformKVectorCrossover& self, const std::vector<std::vector<double>>& parents) {
                 return self.crossover(parents);
             },
             nb::arg("parents"))
        .def("crossover_bits",
             [](UniformKVectorCrossover& self, const std::vector<std::vector<bool>>& parents) {
                 return self.crossover(parents);
             },
             nb::arg("parents"))
        .def("crossover_int",
             [](UniformKVectorCrossover& self, const std::vector<std::vector<int>>& parents) {
                 return self.crossover(parents);
             },
             nb::arg("parents"));

    nb::class_<OrderCrossover, CrossoverOperator>(m, "OrderCrossover")
        .def("__init__", [](OrderCrossover* self, unsigned seed) {
        if (seed == 0) new (self) OrderCrossover{};
        else new (self) OrderCrossover{seed};
    }, nb::arg("seed") = 0u)
        .def("crossover_perm",
             [](OrderCrossover& self, const std::vector<int>& parent1, const std::vector<int>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"));

    nb::class_<PartiallyMappedCrossover, CrossoverOperator>(m, "PartiallyMappedCrossover")
        .def("__init__", [](PartiallyMappedCrossover* self, unsigned seed) {
        if (seed == 0) new (self) PartiallyMappedCrossover{};
        else new (self) PartiallyMappedCrossover{seed};
    }, nb::arg("seed") = 0u)
        .def("crossover_perm",
             [](PartiallyMappedCrossover& self, const std::vector<int>& parent1, const std::vector<int>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"));

    nb::class_<CycleCrossover, CrossoverOperator>(m, "CycleCrossover")
        .def("__init__", [](CycleCrossover* self, unsigned seed) {
        if (seed == 0) new (self) CycleCrossover{};
        else new (self) CycleCrossover{seed};
    }, nb::arg("seed") = 0u)
        .def("crossover_perm",
             [](CycleCrossover& self, const std::vector<int>& parent1, const std::vector<int>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"));

    nb::class_<CutAndCrossfillCrossover, CrossoverOperator>(m, "CutAndCrossfillCrossover")
        .def("__init__", [](CutAndCrossfillCrossover* self, unsigned seed) {
        if (seed == 0) new (self) CutAndCrossfillCrossover{};
        else new (self) CutAndCrossfillCrossover{seed};
    }, nb::arg("seed") = 0u)
        .def("crossover_perm",
             [](CutAndCrossfillCrossover& self, const std::vector<int>& parent1, const std::vector<int>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"));

    nb::class_<EdgeCrossover, CrossoverOperator>(m, "EdgeCrossover")
        .def("__init__", [](EdgeCrossover* self, unsigned seed) {
        if (seed == 0) new (self) EdgeCrossover{};
        else new (self) EdgeCrossover{seed};
    }, nb::arg("seed") = 0u)
        .def("perform_crossover", &EdgeCrossover::performCrossover,
             nb::arg("parent1"), nb::arg("parent2"))
        .def("crossover_perm",
             [](EdgeCrossover& self, const std::vector<int>& parent1, const std::vector<int>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"));

    nb::class_<DiploidRecombination, CrossoverOperator>(m, "DiploidRecombination")
        .def("__init__", [](DiploidRecombination* self, unsigned seed) {
        if (seed == 0) new (self) DiploidRecombination{};
        else new (self) DiploidRecombination{seed};
    }, nb::arg("seed") = 0u)
        .def("crossover_diploid",
             [](DiploidRecombination& self,
                const std::pair<std::vector<bool>, std::vector<bool>>& parent1,
                const std::pair<std::vector<bool>, std::vector<bool>>& parent2) {
                 return self.crossover(parent1, parent2);
             },
             nb::arg("parent1"), nb::arg("parent2"));

    nb::class_<MutationOperator::MutationStats>(m, "MutationStats", "Mutation operator statistics")
        .def(nb::init<>())
        .def_ro("total_mutations", &MutationOperator::MutationStats::totalMutations)
        .def_ro("successful_mutations", &MutationOperator::MutationStats::successfulMutations)
        .def_ro("failed_mutations", &MutationOperator::MutationStats::failedMutations)
        .def_ro("average_perturbation", &MutationOperator::MutationStats::averagePerturbation)
        .def("reset", &MutationOperator::MutationStats::reset);

    nb::class_<MutationOperator>(m, "MutationOperator", "Base class for mutation operators")
        .def_prop_ro("name", &MutationOperator::getName)
        .def("set_seed", &MutationOperator::setSeed, nb::arg("seed"))
        .def("reset_statistics", &MutationOperator::resetStatistics)
        .def_prop_ro("statistics", &MutationOperator::getStatistics,
                               nb::rv_policy::reference_internal);

    nb::class_<GaussianMutation, MutationOperator>(m, "GaussianMutation")
        .def("__init__", [](GaussianMutation* self, unsigned seed) {
        if (seed == 0) new (self) GaussianMutation{};
        else new (self) GaussianMutation{seed};
    }, nb::arg("seed") = 0u)
        .def("mutate_real",
             [](const GaussianMutation& self,
                std::vector<double> chromosome,
                double pm,
                double sigma,
                const std::vector<double>& lower_bounds,
                const std::vector<double>& upper_bounds) {
                 self.mutate(chromosome, pm, sigma, lower_bounds, upper_bounds);
                 return chromosome;
             },
             nb::arg("chromosome"),
             nb::arg("pm"),
             nb::arg("sigma"),
             nb::arg("lower_bounds"),
             nb::arg("upper_bounds"));

    nb::class_<UniformMutation, MutationOperator>(m, "UniformMutation")
        .def("__init__", [](UniformMutation* self, unsigned seed) {
        if (seed == 0) new (self) UniformMutation{};
        else new (self) UniformMutation{seed};
    }, nb::arg("seed") = 0u)
        .def("mutate_real",
             [](const UniformMutation& self,
                std::vector<double> chromosome,
                double pm,
                const std::vector<double>& lower_bounds,
                const std::vector<double>& upper_bounds) {
                 self.mutate(chromosome, pm, lower_bounds, upper_bounds);
                 return chromosome;
             },
             nb::arg("chromosome"),
             nb::arg("pm"),
             nb::arg("lower_bounds"),
             nb::arg("upper_bounds"));

    nb::class_<BitFlipMutation, MutationOperator>(m, "BitFlipMutation")
        .def("__init__", [](BitFlipMutation* self, unsigned seed) {
        if (seed == 0) new (self) BitFlipMutation{};
        else new (self) BitFlipMutation{seed};
    }, nb::arg("seed") = 0u)
        .def("mutate_bits",
             [](const BitFlipMutation& self, std::vector<bool> chromosome, double pm) {
                 self.mutate(chromosome, pm);
                 return chromosome;
             },
             nb::arg("chromosome"),
             nb::arg("pm"))
        .def("mutate_string",
             [](const BitFlipMutation& self, std::string chromosome, double pm) {
                 self.mutate(chromosome, pm);
                 return chromosome;
             },
             nb::arg("chromosome"),
             nb::arg("pm"));

    nb::class_<RandomResettingMutation, MutationOperator>(m, "RandomResettingMutation")
        .def("__init__", [](RandomResettingMutation* self, unsigned seed) {
        if (seed == 0) new (self) RandomResettingMutation{};
        else new (self) RandomResettingMutation{seed};
    }, nb::arg("seed") = 0u)
        .def("mutate_int",
             [](const RandomResettingMutation& self, std::vector<int> chromosome, double pm, int min_val, int max_val) {
                 self.mutate(chromosome, pm, min_val, max_val);
                 return chromosome;
             },
             nb::arg("chromosome"),
             nb::arg("pm"),
             nb::arg("min_val"),
             nb::arg("max_val"));

    nb::class_<CreepMutation, MutationOperator>(m, "CreepMutation")
        .def("__init__", [](CreepMutation* self, unsigned seed) {
        if (seed == 0) new (self) CreepMutation{};
        else new (self) CreepMutation{seed};
    }, nb::arg("seed") = 0u)
        .def("mutate_int",
             [](const CreepMutation& self,
                std::vector<int> chromosome,
                double pm,
                int step_size,
                int min_val,
                int max_val) {
                 self.mutate(chromosome, pm, step_size, min_val, max_val);
                 return chromosome;
             },
             nb::arg("chromosome"),
             nb::arg("pm"),
             nb::arg("step_size"),
             nb::arg("min_val"),
             nb::arg("max_val"));

    nb::class_<SwapMutation, MutationOperator>(m, "SwapMutation")
        .def("__init__", [](SwapMutation* self, unsigned seed) {
        if (seed == 0) new (self) SwapMutation{};
        else new (self) SwapMutation{seed};
    }, nb::arg("seed") = 0u)
        .def("mutate_perm",
             [](const SwapMutation& self, std::vector<int> permutation, double pm) {
                 self.mutate(permutation, pm);
                 return permutation;
             },
             nb::arg("permutation"),
             nb::arg("pm"));

    nb::class_<InversionMutation, MutationOperator>(m, "InversionMutation")
        .def("__init__", [](InversionMutation* self, unsigned seed) {
        if (seed == 0) new (self) InversionMutation{};
        else new (self) InversionMutation{seed};
    }, nb::arg("seed") = 0u)
        .def("mutate_perm",
             [](const InversionMutation& self, std::vector<int> permutation, double pm) {
                 self.mutate(permutation, pm);
                 return permutation;
             },
             nb::arg("permutation"),
             nb::arg("pm"));

    nb::class_<InsertMutation, MutationOperator>(m, "InsertMutation")
        .def("__init__", [](InsertMutation* self, unsigned seed) {
        if (seed == 0) new (self) InsertMutation{};
        else new (self) InsertMutation{seed};
    }, nb::arg("seed") = 0u)
        .def("mutate_perm",
             [](const InsertMutation& self, std::vector<int> permutation, double pm) {
                 self.mutate(permutation, pm);
                 return permutation;
             },
             nb::arg("permutation"),
             nb::arg("pm"));

    nb::class_<ScrambleMutation, MutationOperator>(m, "ScrambleMutation")
        .def("__init__", [](ScrambleMutation* self, unsigned seed) {
        if (seed == 0) new (self) ScrambleMutation{};
        else new (self) ScrambleMutation{seed};
    }, nb::arg("seed") = 0u)
        .def("mutate_perm",
             [](const ScrambleMutation& self, std::vector<int> permutation, double pm) {
                 self.mutate(permutation, pm);
                 return permutation;
             },
             nb::arg("permutation"),
             nb::arg("pm"));

    nb::class_<ListMutation, MutationOperator>(m, "ListMutation")
        .def("__init__", [](ListMutation* self, unsigned seed) {
        if (seed == 0) new (self) ListMutation{};
        else new (self) ListMutation{seed};
    }, nb::arg("seed") = 0u)
        .def("mutate_list",
             [](const ListMutation& self,
                std::vector<int> values,
                double pm_content,
                double pm_size,
                int min_val,
                int max_val,
                std::size_t min_size,
                std::size_t max_size) {
                 self.mutate(values, pm_content, pm_size, min_val, max_val, min_size, max_size);
                 return values;
             },
             nb::arg("values"),
             nb::arg("pm_content"),
             nb::arg("pm_size"),
             nb::arg("min_val"),
             nb::arg("max_val"),
             nb::arg("min_size"),
             nb::arg("max_size"));

    nb::class_<SelfAdaptiveMutation::SelfAdaptiveIndividual>(m, "SelfAdaptiveIndividual",
                                                            "Self-adaptive individual (genes + sigma)")
        .def(nb::init<std::size_t, double>(), nb::arg("size"), nb::arg("initial_sigma"))
        .def_rw("genes", &SelfAdaptiveMutation::SelfAdaptiveIndividual::genes)
        .def_rw("sigma", &SelfAdaptiveMutation::SelfAdaptiveIndividual::sigma);

    nb::class_<SelfAdaptiveMutation, MutationOperator>(m, "SelfAdaptiveMutation")
        .def("__init__", [](SelfAdaptiveMutation* self, unsigned seed) {
        if (seed == 0) new (self) SelfAdaptiveMutation{};
        else new (self) SelfAdaptiveMutation{seed};
    }, nb::arg("seed") = 0u)
        .def("mutate",
             [](const SelfAdaptiveMutation& self,
                SelfAdaptiveMutation::SelfAdaptiveIndividual& individual,
                const std::vector<double>& lower_bounds,
                const std::vector<double>& upper_bounds,
                double tau) {
                 self.mutate(individual, lower_bounds, upper_bounds, tau);
             },
             nb::arg("individual"),
             nb::arg("lower_bounds"),
             nb::arg("upper_bounds"),
             nb::arg("tau") = 0.1);

    // ------------------------------------------------------- Plugin registries
    using CrossoverRegistry = ga::plugin::Registry<CrossoverOperator>;
    nb::class_<CrossoverRegistry>(m, "CrossoverRegistry", "Registry of crossover operator factories")
        .def(nb::init<>())
        .def("register_factory",
             [](CrossoverRegistry& self, const std::string& name, nb::callable factory) {
                 nb::object f = std::move(factory);
                 self.registerFactory(name, [f]() -> std::unique_ptr<CrossoverOperator> {
                     nb::gil_scoped_acquire acquire;
                     nb::object created = f();
                     auto cloned = cloneCrossover(created);
                     if (cloned) return cloned;
                     try {
                         return nb::cast<std::unique_ptr<CrossoverOperator>>(created);
                     } catch (...) {
                         CrossoverOperator* raw = nullptr;
                         if (nb::isinstance<CrossoverOperator>(created)) {
                             raw = nb::cast<CrossoverOperator*>(created);
                         }
                         return std::make_unique<PyCrossoverOperatorWrapper>(created, raw);
                     }
                 });
             },
             nb::arg("name"),
             nb::arg("factory"))
        .def("has", &CrossoverRegistry::has, nb::arg("name"))
        .def("create", &CrossoverRegistry::create, nb::arg("name"))
        .def("names", &CrossoverRegistry::names);

    using MutationRegistry = ga::plugin::Registry<MutationOperator>;
    nb::class_<MutationRegistry>(m, "MutationRegistry", "Registry of mutation operator factories")
        .def(nb::init<>())
        .def("register_factory",
             [](MutationRegistry& self, const std::string& name, nb::callable factory) {
                 nb::object f = std::move(factory);
                 self.registerFactory(name, [f]() -> std::unique_ptr<MutationOperator> {
                     nb::gil_scoped_acquire acquire;
                     nb::object created = f();
                     auto cloned = cloneMutation(created);
                     if (cloned) return cloned;
                     try {
                         return nb::cast<std::unique_ptr<MutationOperator>>(created);
                     } catch (...) {
                         return std::make_unique<PyMutationOperatorWrapper>(created);
                     }
                 });
             },
             nb::arg("name"),
             nb::arg("factory"))
        .def("has", &MutationRegistry::has, nb::arg("name"))
        .def("create", &MutationRegistry::create, nb::arg("name"))
        .def("names", &MutationRegistry::names);

    // ------------------------------------------------------- Operator factories
    m.def("make_gaussian_mutation", &ga::makeGaussianMutation,
          nb::arg("seed") = 0u,
          "Create a Gaussian mutation operator");
    m.def("make_uniform_mutation", &ga::makeUniformMutation,
          nb::arg("seed") = 0u,
          "Create a Uniform mutation operator");
    m.def("make_one_point_crossover", &ga::makeOnePointCrossover,
          nb::arg("seed") = 0u,
          "Create a One-Point crossover operator");
    m.def("make_two_point_crossover", &ga::makeTwoPointCrossover,
          nb::arg("seed") = 0u,
          "Create a Two-Point crossover operator");

    // ------------------------------------------------------- Selection helper APIs
    m.def("selection_tournament_indices",
          [](const std::vector<double>& fitness, std::size_t tournament_size) {
              auto population = fitnessToSelectionPopulation(fitness);
              return TournamentSelection::selectIndices(
                  population, checkedCountToUInt(tournament_size, "tournament_size"));
          },
          nb::arg("fitness"),
          nb::arg("tournament_size") = 3u,
          "Tournament selection helper: returns one winner index from the tournament");

    m.def("selection_roulette_indices",
          [](const std::vector<double>& fitness, std::size_t count) {
              auto population = fitnessToSelectionPopulation(fitness);
              return RouletteWheelSelection::selectIndices(
                  population, checkedCountToUInt(count, "count"));
          },
          nb::arg("fitness"),
          nb::arg("count"),
          "Roulette-wheel selection helper: returns selected indices");

    m.def("selection_rank_indices",
          [](const std::vector<double>& fitness, std::size_t count) {
              auto population = fitnessToSelectionPopulation(fitness);
              // Intentionally route through legacy helper: in this codebase it
              // returns stable original-population indices expected by callers.
              return RankSelectionLegacy(population, checkedCountToUInt(count, "count"));
          },
          nb::arg("fitness"),
          nb::arg("count"),
          "Rank selection helper: returns selected indices");

    m.def("selection_sus_indices",
          [](const std::vector<double>& fitness, std::size_t count) {
              auto population = fitnessToSelectionPopulation(fitness);
              // Intentionally route through legacy helper for index semantics
              // consistent with existing selection utility callers.
              return StochasticUniversalSamplingLegacy(population, checkedCountToUInt(count, "count"));
          },
          nb::arg("fitness"),
          nb::arg("count"),
          "Stochastic universal sampling helper: returns selected indices");

    m.def("selection_elitism_indices",
          [](const std::vector<double>& fitness, std::size_t elite_count) {
              auto population = fitnessToSelectionPopulation(fitness);
              return ElitismSelection::selectIndices(
                  population, checkedCountToUInt(elite_count, "elite_count"));
          },
          nb::arg("fitness"),
          nb::arg("elite_count"),
          "Elitism helper: returns indices of top-fitness individuals");
}
