/**
 * Python bindings for the Genetic Algorithm framework using pybind11.
 *
 * Exposes:
 *  - ga.Config        - Algorithm configuration
 *  - ga.Bounds        - Gene bounds (lower, upper)
 *  - ga.Result        - Run results
 *  - ga.GeneticAlgorithm - Main GA class
 *  - Operator factories: make_gaussian_mutation, make_uniform_mutation,
 *                        make_one_point_crossover, make_two_point_crossover
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <random>
#include <stdexcept>
#include <thread>

#include "ga/config.hpp"
#include "ga/genetic_algorithm.hpp"
#include "ga/api/builder.hpp"
#include "ga/api/optimizer.hpp"
#include "ga/adaptive/adaptive_policy.hpp"
#include "ga/algorithms/moea/nsga2.hpp"
#include "ga/checkpoint/checkpoint.hpp"
#include "ga/constraints/constraints.hpp"
#include "ga/coevolution/coevolution.hpp"
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
#include "ga/moea/nsga3.hpp"
#include "ga/moea/mo_cmaes.hpp"
#include "ga/moea/spea2.hpp"
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
#include "crossover/base_crossover.h"
#include "selection-operator/tournament_selection.h"
#include "selection-operator/roulette_wheel_selection.h"
#include "selection-operator/rank_selection.h"
#include "selection-operator/stochastic_universal_sampling.h"
#include "selection-operator/elitism_selection.h"

namespace py = pybind11;

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

static std::vector<ga::api::Optimizer::Objective> pyObjectivesToCpp(const py::iterable& objectiveCallables) {
    std::vector<ga::api::Optimizer::Objective> objectives;
    for (py::handle h : objectiveCallables) {
        py::object obj = py::reinterpret_borrow<py::object>(h);
        objectives.emplace_back([obj](const std::vector<double>& genes) {
            py::gil_scoped_acquire acquire;
            return obj(genes).cast<double>();
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

PYBIND11_MODULE(ga, m) {
    m.doc() = "Genetic Algorithm framework — C++ core with Python bindings";

    // ------------------------------------------------------------------ Bounds
    py::class_<ga::Bounds>(m, "Bounds", "Gene value bounds [lower, upper]")
        .def(py::init<>())
        .def(py::init([](double lo, double hi){
            ga::Bounds b; b.lower = lo; b.upper = hi; return b;
        }), py::arg("lower"), py::arg("upper"))
        .def_readwrite("lower", &ga::Bounds::lower)
        .def_readwrite("upper", &ga::Bounds::upper)
        .def("__repr__", [](const ga::Bounds& b){
            return "Bounds(lower=" + std::to_string(b.lower) + ", upper=" + std::to_string(b.upper) + ")";
        });

    // ------------------------------------------------------------------ Config
    py::class_<ga::Config>(m, "Config", "Genetic Algorithm configuration")
        .def(py::init<>())
        .def_readwrite("population_size", &ga::Config::populationSize,  "Number of individuals")
        .def_readwrite("generations",     &ga::Config::generations,      "Number of generations")
        .def_readwrite("dimension",       &ga::Config::dimension,        "Gene vector length")
        .def_readwrite("crossover_rate",  &ga::Config::crossoverRate,    "Crossover probability [0,1]")
        .def_readwrite("mutation_rate",   &ga::Config::mutationRate,     "Per-gene mutation probability [0,1]")
        .def_readwrite("bounds",          &ga::Config::bounds,           "Gene search bounds")
        .def_readwrite("elite_ratio",     &ga::Config::eliteRatio,       "Elite fraction preserved each gen [0,1]")
        .def_readwrite("seed",            &ga::Config::seed,             "RNG seed (0 = random)")
        .def("__repr__", [](const ga::Config& c){
            return "<Config pop=" + std::to_string(c.populationSize)
                 + " gen=" + std::to_string(c.generations)
                 + " dim=" + std::to_string(c.dimension) + ">";
        });

    // ------------------------------------------------------------------ Result
    py::class_<ga::Result>(m, "Result", "Results returned by GeneticAlgorithm.run()")
        .def(py::init<>())
        .def_readonly("best_genes",    &ga::Result::bestGenes,    "Best gene vector found")
        .def_readonly("best_fitness",  &ga::Result::bestFitness,  "Fitness of the best individual")
        .def_readonly("best_history",  &ga::Result::bestHistory,  "Best fitness per generation")
        .def_readonly("avg_history",   &ga::Result::avgHistory,   "Average fitness per generation")
        .def("__repr__", [](const ga::Result& r){
            return "<Result best_fitness=" + std::to_string(r.bestFitness) + ">";
        });

    py::class_<ga::core::OptimizationResult>(m, "OptimizationResult", "Generic optimization result container")
        .def(py::init<>())
        .def_readwrite("best_solution", &ga::core::OptimizationResult::bestSolution)
        .def_readwrite("best_fitness", &ga::core::OptimizationResult::bestFitness)
        .def_readwrite("best_history", &ga::core::OptimizationResult::bestHistory)
        .def_readwrite("avg_history", &ga::core::OptimizationResult::avgHistory)
        .def_readwrite("pareto_objectives", &ga::core::OptimizationResult::paretoObjectives)
        .def_readwrite("pareto_genes", &ga::core::OptimizationResult::paretoGenes)
        .def_readwrite("evaluations", &ga::core::OptimizationResult::evaluations)
        .def_readwrite("generations", &ga::core::OptimizationResult::generations);

    // ---------------------------------------------------------- Core abstractions
    py::class_<ga::Evaluation>(m, "Evaluation", "Evaluation/objective record")
        .def(py::init<>())
        .def_readwrite("objectives", &ga::Evaluation::objectives)
        .def_readwrite("feasible", &ga::Evaluation::feasible)
        .def_readwrite("penalty", &ga::Evaluation::penalty);

    py::class_<ga::Individual>(m, "Individual", "Generic individual container")
        .def(py::init<>())
        .def_readwrite("evaluation", &ga::Individual::evaluation)
        .def_readwrite("age", &ga::Individual::age);

    py::class_<ga::IGenome>(m, "IGenome", "Genome interface")
        .def("encoding_name", &ga::IGenome::encodingName);

    // ------------------------------------------------------------- Representations
    py::class_<ga::representations::VectorGenome<double>, ga::IGenome>(m, "VectorGenome")
        .def(py::init<>())
        .def(py::init<std::vector<double>>(), py::arg("genes"))
        .def_readwrite("genes", &ga::representations::VectorGenome<double>::genes)
        .def("encoding_name", &ga::representations::VectorGenome<double>::encodingName);

    py::class_<ga::representations::BitsetGenome, ga::IGenome>(m, "BitsetGenome")
        .def(py::init<>())
        .def(py::init<std::size_t, bool>(), py::arg("size"), py::arg("fill") = false)
        .def(py::init<std::vector<bool>>(), py::arg("bits"))
        .def_readwrite("bits", &ga::representations::BitsetGenome::bits)
        .def("size", &ga::representations::BitsetGenome::size)
        .def("hamming_distance", &ga::representations::BitsetGenome::hammingDistance, py::arg("other"))
        .def("popcount", &ga::representations::BitsetGenome::popcount)
        .def("encoding_name", &ga::representations::BitsetGenome::encodingName);

    py::class_<ga::representations::PermutationGenome, ga::IGenome>(m, "PermutationGenome")
        .def(py::init<>())
        .def(py::init<std::size_t>(), py::arg("size"))
        .def(py::init<std::vector<int>>(), py::arg("order"))
        .def_readwrite("order", &ga::representations::PermutationGenome::order)
        .def("size", &ga::representations::PermutationGenome::size)
        .def("is_valid", &ga::representations::PermutationGenome::isValid)
        .def("position_of", &ga::representations::PermutationGenome::positionOf, py::arg("value"))
        .def_static("random", [](std::size_t n, unsigned seed) {
            std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
            return ga::representations::PermutationGenome::random(n, rng);
        }, py::arg("size"), py::arg("seed") = 0u)
        .def("encoding_name", &ga::representations::PermutationGenome::encodingName);

    py::class_<ga::representations::SetGenome, ga::IGenome>(m, "SetGenome")
        .def(py::init<>())
        .def(py::init<std::set<int>>(), py::arg("values"))
        .def_readwrite("values", &ga::representations::SetGenome::values)
        .def("encoding_name", &ga::representations::SetGenome::encodingName);

    py::class_<ga::representations::MapGenome, ga::IGenome>(m, "MapGenome")
        .def(py::init<>())
        .def(py::init<std::unordered_map<std::string, double>>(), py::arg("values"))
        .def_readwrite("values", &ga::representations::MapGenome::values)
        .def("encoding_name", &ga::representations::MapGenome::encodingName);

    py::class_<ga::representations::NdArrayGenome, ga::IGenome>(m, "NdArrayGenome")
        .def(py::init<>())
        .def(py::init<std::size_t, std::size_t>(), py::arg("rows"), py::arg("cols"))
        .def_readwrite("rows", &ga::representations::NdArrayGenome::rows)
        .def_readwrite("cols", &ga::representations::NdArrayGenome::cols)
        .def_readwrite("data", &ga::representations::NdArrayGenome::data)
        .def("get", [](const ga::representations::NdArrayGenome& self, std::size_t r, std::size_t c) {
            return self.at(r, c);
        }, py::arg("row"), py::arg("col"))
        .def("set", [](ga::representations::NdArrayGenome& self, std::size_t r, std::size_t c, double value) {
            self.at(r, c) = value;
        }, py::arg("row"), py::arg("col"), py::arg("value"))
        .def("encoding_name", &ga::representations::NdArrayGenome::encodingName);

    py::enum_<ga::gp::ValueType>(m, "ValueType")
        .value("any", ga::gp::ValueType::Any)
        .value("bool", ga::gp::ValueType::Bool)
        .value("int", ga::gp::ValueType::Int)
        .value("double", ga::gp::ValueType::Double);

    py::class_<ga::gp::Signature>(m, "Signature")
        .def(py::init<>())
        .def_readwrite("return_type", &ga::gp::Signature::returnType)
        .def_readwrite("arg_types", &ga::gp::Signature::argTypes);

    py::class_<ga::gp::Primitive>(m, "Primitive")
        .def(py::init<>())
        .def_readwrite("name", &ga::gp::Primitive::name)
        .def_readwrite("signature", &ga::gp::Primitive::signature)
        .def_readwrite("is_terminal", &ga::gp::Primitive::isTerminal);

    py::class_<ga::gp::Node>(m, "Node")
        .def(py::init<>())
        .def(py::init<std::string, ga::gp::ValueType>(), py::arg("symbol"), py::arg("return_type"))
        .def_readwrite("symbol", &ga::gp::Node::symbol)
        .def_readwrite("return_type", &ga::gp::Node::returnType)
        .def("size", &ga::gp::Node::size)
        .def("child_count", [](const ga::gp::Node& self) { return self.children.size(); })
        .def("add_child", [](ga::gp::Node& self, const ga::gp::Node& child) {
            self.children.push_back(child.clone());
        });

    py::class_<ga::representations::TreeGenome, ga::IGenome>(m, "TreeGenome")
        .def(py::init<>())
        .def(py::init([](const ga::gp::Node& root) {
            return ga::representations::TreeGenome(root.clone());
        }), py::arg("root"))
        .def("has_root", [](const ga::representations::TreeGenome& self) { return static_cast<bool>(self.root); })
        .def("set_root", [](ga::representations::TreeGenome& self, const ga::gp::Node& root) {
            self.root = root.clone();
        }, py::arg("root"))
        .def("root", [](const ga::representations::TreeGenome& self) -> py::object {
            if (!self.root) {
                return py::none();
            }
            return py::cast(self.root.get(), py::return_value_policy::reference);
        })
        .def("encoding_name", &ga::representations::TreeGenome::encodingName);

    py::class_<ga::gp::TreeBuilder>(m, "TreeBuilder")
        .def(py::init<std::vector<ga::gp::Primitive>>(), py::arg("primitives"))
        .def("grow", [](const ga::gp::TreeBuilder& self,
                        std::size_t maxDepth,
                        ga::gp::ValueType targetType,
                        bool stronglyTyped,
                        unsigned seed) {
            std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
            return self.grow(maxDepth, targetType, stronglyTyped, rng);
        }, py::arg("max_depth"), py::arg("target_type") = ga::gp::ValueType::Any,
           py::arg("strongly_typed") = false, py::arg("seed") = 0u);

    py::class_<ga::gp::ADFPool>(m, "ADFPool")
        .def(py::init<>())
        .def("put", &ga::gp::ADFPool::put, py::arg("name"), py::arg("root"))
        .def("has", &ga::gp::ADFPool::has, py::arg("name"))
        .def("get", [](const ga::gp::ADFPool& self, const std::string& name) -> const ga::gp::Node& {
            return self.get(name);
        }, py::arg("name"), py::return_value_policy::reference_internal)
        .def("size", &ga::gp::ADFPool::size);

    // -------------------------------------------------------- GeneticAlgorithm
    py::class_<ga::GeneticAlgorithm>(m, "GeneticAlgorithm",
        R"(Main Genetic Algorithm class.

        Example
        -------
        >>> import ga
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
        .def(py::init<const ga::Config&>(), py::arg("config"))
        .def("run", &ga::GeneticAlgorithm::run, py::arg("fitness"),
             "Run the GA with the given fitness callable (list[float] -> float). Higher is better.")
        .def("set_mutation_operator", &ga::GeneticAlgorithm::setMutationOperator,
             py::arg("op"), "Set a custom mutation operator")
        .def("set_crossover_operator", &ga::GeneticAlgorithm::setCrossoverOperator,
             py::arg("op"), "Set a custom crossover operator")
        .def("config", &ga::GeneticAlgorithm::config,
             py::return_value_policy::reference_internal, "Return the current Config");

    // ------------------------------------------------------------------- NSGA-II
    py::class_<ga::moea::Nsga2Config>(m, "Nsga2Config", "NSGA-II configuration")
        .def(py::init<>())
        .def_readwrite("population_size", &ga::moea::Nsga2Config::populationSize)
        .def_readwrite("generations", &ga::moea::Nsga2Config::generations)
        .def_readwrite("seed", &ga::moea::Nsga2Config::seed);

    py::class_<ga::moea::Nsga2>(m, "Nsga2", "NSGA-II utility methods for objective-space operations")
        .def(py::init<const ga::moea::Nsga2Config&>(), py::arg("config") = ga::moea::Nsga2Config{})
        .def("non_dominated_sort_objectives",
             [](const ga::moea::Nsga2& self, const std::vector<std::vector<double>>& objectiveMatrix) {
                 auto population = objectivesToIndividuals(objectiveMatrix);
                 return self.nonDominatedSort(population);
             },
             py::arg("objectives"),
             "Run non-dominated sorting on objective vectors (minimization)")
        .def("crowding_distance_objectives",
             [](const ga::moea::Nsga2& self,
                const std::vector<std::vector<double>>& objectiveMatrix,
                const std::vector<std::size_t>& front) {
                 auto population = objectivesToIndividuals(objectiveMatrix);
                 return self.crowdingDistance(population, front);
             },
             py::arg("objectives"),
             py::arg("front"),
             "Compute crowding distance for one front over objective vectors");

    m.def("nsga2_non_dominated_sort",
          [](const std::vector<std::vector<double>>& objectiveMatrix) {
              ga::moea::Nsga2 nsga2;
              auto population = objectivesToIndividuals(objectiveMatrix);
              return nsga2.nonDominatedSort(population);
          },
          py::arg("objectives"),
          "Convenience function: non-dominated sorting in objective space");

    m.def("nsga2_crowding_distance",
          [](const std::vector<std::vector<double>>& objectiveMatrix,
             const std::vector<std::size_t>& front) {
              ga::moea::Nsga2 nsga2;
              auto population = objectivesToIndividuals(objectiveMatrix);
              return nsga2.crowdingDistance(population, front);
          },
          py::arg("objectives"),
          py::arg("front"),
          "Convenience function: crowding distance in objective space");

    py::class_<ga::moea::Spea2>(m, "Spea2", "SPEA2 objective-space utilities")
        .def(py::init<>())
        .def("strength_fitness_objectives",
             [](const ga::moea::Spea2& self, const std::vector<std::vector<double>>& objectiveMatrix) {
                 auto population = objectivesToIndividuals(objectiveMatrix);
                 return self.strengthFitness(population);
             },
             py::arg("objectives"),
             "Compute SPEA2 strength fitness values in objective space (lower is better)")
        .def("environmental_select_objectives",
             [](const ga::moea::Spea2& self,
                const std::vector<std::vector<double>>& objectiveMatrix,
                std::size_t targetSize) {
                 auto population = objectivesToIndividuals(objectiveMatrix);
                 auto selected = self.environmentalSelect(population, targetSize);
                 return individualsToObjectives(selected);
             },
             py::arg("objectives"),
             py::arg("target_size"),
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
             py::arg("objectives"),
             py::arg("target_size"),
             "Select indices into the input objective vectors using SPEA2");

    m.def("spea2_strength_fitness",
          [](const std::vector<std::vector<double>>& objectiveMatrix) {
              ga::moea::Spea2 spea2;
              auto population = objectivesToIndividuals(objectiveMatrix);
              return spea2.strengthFitness(population);
          },
          py::arg("objectives"),
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
          py::arg("objectives"),
          py::arg("target_size"),
          "Convenience function: SPEA2 environmental selection indices");

    // ------------------------------------------------------------------- NSGA-III
    py::class_<ga::moea::Nsga3>(m, "Nsga3", "NSGA-III utility methods for objective-space operations")
        .def(py::init<ga::moea::Nsga2Config>(), py::arg("config") = ga::moea::Nsga2Config{})
        .def_static("generate_reference_points",
                    &ga::moea::Nsga3::generateDasDennisReferencePoints,
                    py::arg("objective_count"),
                    py::arg("divisions"),
                    "Generate Das-Dennis reference points")
        .def("non_dominated_sort_objectives",
             [](const ga::moea::Nsga3& self,
                const std::vector<std::vector<double>>& objectiveMatrix) {
                 auto population = objectivesToIndividuals(objectiveMatrix);
                 return self.nonDominatedSort(population);
             },
             py::arg("objectives"),
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
             py::arg("objectives"),
             py::arg("target_size"),
             py::arg("reference_points"),
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
             py::arg("objectives"),
             py::arg("target_size"),
             py::arg("reference_points"),
             "Select indices into the input objective vectors using NSGA-III niching");

    m.def("nsga3_reference_points",
          &ga::moea::Nsga3::generateDasDennisReferencePoints,
          py::arg("objective_count"),
          py::arg("divisions"),
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
          py::arg("objectives"),
          py::arg("target_size"),
          py::arg("reference_points"),
          "Convenience function: NSGA-III environmental selection over objective vectors");

    // -------------------------------------------------------------- Checkpoint API
    py::class_<ga::checkpoint::CheckpointState>(m, "CheckpointState", "Checkpoint serialization state")
        .def(py::init<>())
        .def_readwrite("config", &ga::checkpoint::CheckpointState::config)
        .def_readwrite("result", &ga::checkpoint::CheckpointState::result)
        .def_readwrite("generation", &ga::checkpoint::CheckpointState::generation)
        .def_readwrite("rng_state", &ga::checkpoint::CheckpointState::rngState);

    m.def("checkpoint_save_json",
          [](const std::string& path, const ga::checkpoint::CheckpointState& state) {
              ga::checkpoint::CheckpointManager::saveJson(path, state);
          },
          py::arg("path"),
          py::arg("state"),
          "Save checkpoint state as JSON");

    m.def("checkpoint_load_json",
          [](const std::string& path) {
              return ga::checkpoint::CheckpointManager::loadJson(path);
          },
          py::arg("path"),
          "Load checkpoint state from JSON");

    m.def("checkpoint_save_binary",
          [](const std::string& path, const ga::checkpoint::CheckpointState& state) {
              ga::checkpoint::CheckpointManager::saveBinary(path, state);
          },
          py::arg("path"),
          py::arg("state"),
          "Save checkpoint state as binary");

    m.def("checkpoint_load_binary",
          [](const std::string& path) {
              return ga::checkpoint::CheckpointManager::loadBinary(path);
          },
          py::arg("path"),
          "Load checkpoint state from binary");

    // -------------------------------------------------------------------- Optimizer API
    py::class_<ga::api::Optimizer::MultiObjectiveResult>(m, "MultiObjectiveResult",
                                                          "Multi-objective optimizer result")
        .def(py::init<>())
        .def_readwrite("pareto_genes", &ga::api::Optimizer::MultiObjectiveResult::paretoGenes)
        .def_readwrite("pareto_objectives", &ga::api::Optimizer::MultiObjectiveResult::paretoObjectives);

    py::class_<ga::api::Optimizer>(m, "Optimizer", "High-level optimizer facade")
        .def(py::init<>())
        .def("with_config", &ga::api::Optimizer::withConfig, py::arg("config"),
             py::return_value_policy::reference_internal)
        .def("with_threads", &ga::api::Optimizer::withThreads, py::arg("threads"),
             py::return_value_policy::reference_internal)
        .def("with_seed", &ga::api::Optimizer::withSeed, py::arg("seed"),
             py::return_value_policy::reference_internal)
        .def("optimize", &ga::api::Optimizer::optimize, py::arg("objective"))
        .def("optimize_multi_objective_nsga2",
             [](const ga::api::Optimizer& self, const py::iterable& objectiveCallables,
                std::size_t populationSize, std::size_t generations) {
                 auto objectives = pyObjectivesToCpp(objectiveCallables);
                 return self.optimizeMultiObjectiveNsga2(objectives, populationSize, generations);
             },
             py::arg("objectives"), py::arg("population_size") = 80, py::arg("generations") = 80)
        .def("optimize_multi_objective_nsga3",
             [](const ga::api::Optimizer& self, const py::iterable& objectiveCallables,
                std::size_t populationSize, std::size_t generations, std::size_t referenceDivisions) {
                 auto objectives = pyObjectivesToCpp(objectiveCallables);
                 return self.optimizeMultiObjectiveNsga3(
                     objectives, populationSize, generations, referenceDivisions);
             },
             py::arg("objectives"), py::arg("population_size") = 80, py::arg("generations") = 80,
             py::arg("reference_divisions") = 8);

    py::class_<ga::api::OptimizerBuilder>(m, "OptimizerBuilder", "Fluent optimizer builder")
        .def(py::init<>())
        .def("dimension", &ga::api::OptimizerBuilder::dimension, py::arg("dimension"),
             py::return_value_policy::reference_internal)
        .def("bounds", &ga::api::OptimizerBuilder::bounds, py::arg("lower"), py::arg("upper"),
             py::return_value_policy::reference_internal)
        .def("population_size", &ga::api::OptimizerBuilder::populationSize, py::arg("population_size"),
             py::return_value_policy::reference_internal)
        .def("generations", &ga::api::OptimizerBuilder::generations, py::arg("generations"),
             py::return_value_policy::reference_internal)
        .def("seed", &ga::api::OptimizerBuilder::seed, py::arg("seed"),
             py::return_value_policy::reference_internal)
        .def("crossover_rate", &ga::api::OptimizerBuilder::crossoverRate, py::arg("crossover_rate"),
             py::return_value_policy::reference_internal)
        .def("mutation_rate", &ga::api::OptimizerBuilder::mutationRate, py::arg("mutation_rate"),
             py::return_value_policy::reference_internal)
        .def("elite_ratio", &ga::api::OptimizerBuilder::eliteRatio, py::arg("elite_ratio"),
             py::return_value_policy::reference_internal)
        .def("threads", &ga::api::OptimizerBuilder::threads, py::arg("threads"),
             py::return_value_policy::reference_internal)
        .def("build", &ga::api::OptimizerBuilder::build);

    // ------------------------------------------------------------ ES / CMA-ES / MO-CMA-ES
    py::class_<ga::es::EvolutionStrategyConfig>(m, "EvolutionStrategyConfig")
        .def(py::init<>())
        .def_readwrite("mu", &ga::es::EvolutionStrategyConfig::mu)
        .def_readwrite("lambda_", &ga::es::EvolutionStrategyConfig::lambda)
        .def_readwrite("generations", &ga::es::EvolutionStrategyConfig::generations)
        .def_readwrite("dimension", &ga::es::EvolutionStrategyConfig::dimension)
        .def_readwrite("sigma", &ga::es::EvolutionStrategyConfig::sigma)
        .def_readwrite("lower", &ga::es::EvolutionStrategyConfig::lower)
        .def_readwrite("upper", &ga::es::EvolutionStrategyConfig::upper)
        .def_readwrite("plus_strategy", &ga::es::EvolutionStrategyConfig::plusStrategy)
        .def_readwrite("seed", &ga::es::EvolutionStrategyConfig::seed);

    py::class_<ga::es::EvolutionStrategyResult>(m, "EvolutionStrategyResult")
        .def(py::init<>())
        .def_readwrite("best", &ga::es::EvolutionStrategyResult::best)
        .def_readwrite("best_fitness", &ga::es::EvolutionStrategyResult::bestFitness)
        .def_readwrite("best_history", &ga::es::EvolutionStrategyResult::bestHistory);

    py::class_<ga::es::EvolutionStrategy>(m, "EvolutionStrategy")
        .def(py::init<ga::es::EvolutionStrategyConfig>(), py::arg("config"))
        .def("run", &ga::es::EvolutionStrategy::run, py::arg("fitness"));

    py::class_<ga::es::CmaEsConfig>(m, "CmaEsConfig")
        .def(py::init<>())
        .def_readwrite("population_size", &ga::es::CmaEsConfig::populationSize)
        .def_readwrite("generations", &ga::es::CmaEsConfig::generations)
        .def_readwrite("dimension", &ga::es::CmaEsConfig::dimension)
        .def_readwrite("lower", &ga::es::CmaEsConfig::lower)
        .def_readwrite("upper", &ga::es::CmaEsConfig::upper)
        .def_readwrite("sigma", &ga::es::CmaEsConfig::sigma)
        .def_readwrite("seed", &ga::es::CmaEsConfig::seed);

    py::class_<ga::es::CmaEsResult>(m, "CmaEsResult")
        .def(py::init<>())
        .def_readwrite("best", &ga::es::CmaEsResult::best)
        .def_readwrite("best_fitness", &ga::es::CmaEsResult::bestFitness)
        .def_readwrite("history", &ga::es::CmaEsResult::history);

    py::class_<ga::es::DiagonalCmaEs>(m, "DiagonalCmaEs")
        .def(py::init<ga::es::CmaEsConfig>(), py::arg("config"))
        .def("run", &ga::es::DiagonalCmaEs::run, py::arg("fitness"));

    py::class_<ga::moea::MoCmaEsConfig>(m, "MoCmaEsConfig")
        .def(py::init<>())
        .def_readwrite("cma", &ga::moea::MoCmaEsConfig::cma)
        .def_readwrite("weights", &ga::moea::MoCmaEsConfig::weights);

    py::class_<ga::moea::MoCmaEsResult>(m, "MoCmaEsResult")
        .def(py::init<>())
        .def_readwrite("best", &ga::moea::MoCmaEsResult::best)
        .def_readwrite("objectives", &ga::moea::MoCmaEsResult::objectives)
        .def_readwrite("weighted_fitness", &ga::moea::MoCmaEsResult::weightedFitness);

    py::class_<ga::moea::MoCmaEs>(m, "MoCmaEs")
        .def(py::init<ga::moea::MoCmaEsConfig>(), py::arg("config"))
        .def("run", &ga::moea::MoCmaEs::run, py::arg("objective"));

    // ----------------------------------------------------- Constraints and adaptation
    py::class_<ga::constraints::ConstraintSet>(m, "ConstraintSet")
        .def(py::init<>())
        .def("add_hard_constraint", [](ga::constraints::ConstraintSet& self, py::function fn) {
            self.hard.emplace_back([fn](const std::vector<double>& genes) {
                py::gil_scoped_acquire acquire;
                return fn(genes).cast<bool>();
            });
        }, py::arg("constraint"))
        .def("add_soft_penalty", [](ga::constraints::ConstraintSet& self, py::function fn) {
            self.soft.emplace_back([fn](const std::vector<double>& genes) {
                py::gil_scoped_acquire acquire;
                return fn(genes).cast<double>();
            });
        }, py::arg("penalty"))
        .def("add_repair", [](ga::constraints::ConstraintSet& self, py::function fn) {
            self.repairs.emplace_back([fn](std::vector<double>& genes) {
                py::gil_scoped_acquire acquire;
                py::object out = fn(genes);
                if (!out.is_none()) {
                    genes = out.cast<std::vector<double>>();
                }
            });
        }, py::arg("repair"))
        .def("clear", [](ga::constraints::ConstraintSet& self) {
            self.hard.clear();
            self.soft.clear();
            self.repairs.clear();
        });

    m.def("is_feasible", &ga::constraints::isFeasible, py::arg("genes"), py::arg("constraint_set"));
    m.def("total_penalty", &ga::constraints::totalPenalty, py::arg("genes"), py::arg("constraint_set"));
    m.def("apply_repairs", [](std::vector<double> genes, const ga::constraints::ConstraintSet& set) {
        ga::constraints::applyRepairs(genes, set);
        return genes;
    }, py::arg("genes"), py::arg("constraint_set"));
    m.def("penalized_fitness", &ga::constraints::penalizedFitness,
          py::arg("base_fitness"), py::arg("genes"), py::arg("constraint_set"),
          py::arg("infeasible_penalty") = 1e6);

    py::class_<ga::adaptive::AdaptiveRates>(m, "AdaptiveRates")
        .def(py::init<>())
        .def_readwrite("mutation_rate", &ga::adaptive::AdaptiveRates::mutationRate)
        .def_readwrite("crossover_rate", &ga::adaptive::AdaptiveRates::crossoverRate);

    py::class_<ga::adaptive::AdaptiveRateController>(m, "AdaptiveRateController")
        .def(py::init<double, double, double, double>(),
             py::arg("min_mutation") = 0.001, py::arg("max_mutation") = 0.6,
             py::arg("min_crossover") = 0.4, py::arg("max_crossover") = 0.95)
        .def("update", &ga::adaptive::AdaptiveRateController::update, py::arg("current"),
             py::arg("diversity"), py::arg("best_improvement"));

    // ---------------------------------------------------------- Hybrid / Coevolution
    py::class_<ga::hybrid::HybridOptimizer>(m, "HybridOptimizer")
        .def(py::init<ga::Config>(), py::arg("config"))
        .def("run",
             [](const ga::hybrid::HybridOptimizer& self,
                const ga::Fitness& fitness,
                py::object localSearch,
                std::size_t localSearchRestarts) {
                 ga::hybrid::HybridOptimizer::LocalSearch ls;
                 if (!localSearch.is_none()) {
                     ls = [localSearch](std::vector<double>& genes) {
                         py::gil_scoped_acquire acquire;
                         py::object out = localSearch(genes);
                         if (!out.is_none()) {
                             genes = out.cast<std::vector<double>>();
                         }
                     };
                 }
                 return self.run(fitness, ls, localSearchRestarts);
             },
             py::arg("fitness"),
             py::arg("local_search") = py::none(),
             py::arg("local_search_restarts") = 5);

    py::class_<ga::coevolution::CoevolutionConfig>(m, "CoevolutionConfig")
        .def(py::init<>())
        .def_readwrite("generations", &ga::coevolution::CoevolutionConfig::generations)
        .def_readwrite("seed", &ga::coevolution::CoevolutionConfig::seed);

    py::class_<ga::coevolution::CoevolutionEngine>(m, "CoevolutionEngine")
        .def(py::init<ga::coevolution::CoevolutionConfig>(), py::arg("config"))
        .def("run",
             [](const ga::coevolution::CoevolutionEngine& self,
                ga::coevolution::CoevolutionEngine::Populations populations,
                py::object evaluate,
                py::object reproduce) {
                 ga::coevolution::CoevolutionEngine::EvaluateFn evalFn;
                 if (!evaluate.is_none()) {
                     evalFn = [evaluate](ga::coevolution::CoevolutionEngine::Populations& pops) {
                         py::gil_scoped_acquire acquire;
                         evaluate(pops);
                     };
                 }
                 ga::coevolution::CoevolutionEngine::ReproduceFn repFn;
                 if (!reproduce.is_none()) {
                     repFn = [reproduce](ga::coevolution::CoevolutionEngine::Populations& pops, std::mt19937&) {
                         py::gil_scoped_acquire acquire;
                         reproduce(pops);
                     };
                 }
                 return self.run(std::move(populations), evalFn, repFn);
             },
             py::arg("populations"),
             py::arg("evaluate") = py::none(),
             py::arg("reproduce") = py::none());

    // ------------------------------------------------------ Tracking / visualization
    py::class_<ga::tracking::ExperimentTracker>(m, "ExperimentTracker")
        .def(py::init<std::string>(), py::arg("run_id"))
        .def("write_config", &ga::tracking::ExperimentTracker::writeConfig,
             py::arg("config"), py::arg("path"))
        .def("write_history_csv", &ga::tracking::ExperimentTracker::writeHistoryCSV,
             py::arg("result"), py::arg("path"))
        .def("write_best_solution_csv", &ga::tracking::ExperimentTracker::writeBestSolutionCSV,
             py::arg("result"), py::arg("path"));

    m.def("export_fitness_curve_csv", &ga::visualization::exportFitnessCurveCSV,
          py::arg("best"), py::arg("avg"), py::arg("path"));
    m.def("export_pareto_front_csv", &ga::visualization::exportParetoFrontCSV,
          py::arg("objectives"), py::arg("path"));
    m.def("export_diversity_csv", &ga::visualization::exportDiversityCSV,
          py::arg("diversity"), py::arg("path"));

    // ------------------------------------------------------- Operator factories
    m.def("make_gaussian_mutation", &ga::makeGaussianMutation,
          py::arg("seed") = 0u,
          "Create a Gaussian mutation operator");
    m.def("make_uniform_mutation", &ga::makeUniformMutation,
          py::arg("seed") = 0u,
          "Create a Uniform mutation operator");
    m.def("make_one_point_crossover", &ga::makeOnePointCrossover,
          py::arg("seed") = 0u,
          "Create a One-Point crossover operator");
    m.def("make_two_point_crossover", &ga::makeTwoPointCrossover,
          py::arg("seed") = 0u,
          "Create a Two-Point crossover operator");

    // ------------------------------------------------------- Selection helper APIs
    m.def("selection_tournament_indices",
          [](const std::vector<double>& fitness, std::size_t tournamentSize) {
              auto population = fitnessToSelectionPopulation(fitness);
              return TournamentSelection::selectIndices(
                  population, static_cast<unsigned int>(tournamentSize));
          },
          py::arg("fitness"),
          py::arg("tournament_size") = 3u,
          "Tournament selection helper: returns one winner index from the tournament");

    m.def("selection_roulette_indices",
          [](const std::vector<double>& fitness, std::size_t count) {
              auto population = fitnessToSelectionPopulation(fitness);
              return RouletteWheelSelection::selectIndices(
                  population, static_cast<unsigned int>(count));
          },
          py::arg("fitness"),
          py::arg("count"),
          "Roulette-wheel selection helper: returns selected indices");

    m.def("selection_rank_indices",
          [](const std::vector<double>& fitness, std::size_t count) {
              auto population = fitnessToSelectionPopulation(fitness);
              return RankSelectionLegacy(population, static_cast<unsigned int>(count));
          },
          py::arg("fitness"),
          py::arg("count"),
          "Rank selection helper: returns selected indices");

    m.def("selection_sus_indices",
          [](const std::vector<double>& fitness, std::size_t count) {
              auto population = fitnessToSelectionPopulation(fitness);
              return StochasticUniversalSamplingLegacy(population, static_cast<unsigned int>(count));
          },
          py::arg("fitness"),
          py::arg("count"),
          "Stochastic universal sampling helper: returns selected indices");

    m.def("selection_elitism_indices",
          [](const std::vector<double>& fitness, std::size_t eliteCount) {
              auto population = fitnessToSelectionPopulation(fitness);
              return ElitismSelection::selectIndices(
                  population, static_cast<unsigned int>(eliteCount));
          },
          py::arg("fitness"),
          py::arg("elite_count"),
          "Elitism helper: returns indices of top-fitness individuals");
}
