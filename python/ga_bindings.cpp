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

#include <stdexcept>

#include "ga/config.hpp"
#include "ga/genetic_algorithm.hpp"
#include "ga/algorithms/moea/nsga2.hpp"
#include "ga/checkpoint/checkpoint.hpp"
#include "ga/moea/nsga3.hpp"

// Full type definitions needed by pybind11 for operator ownership transfer
#include "mutation/base_mutation.h"
#include "crossover/base_crossover.h"

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
}
