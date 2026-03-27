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

#include "ga/config.hpp"
#include "ga/genetic_algorithm.hpp"

// Full type definitions needed by pybind11 for operator ownership transfer
#include "mutation/base_mutation.h"
#include "crossover/base_crossover.h"

namespace py = pybind11;

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
