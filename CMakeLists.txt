cmake_minimum_required(VERSION 3.16)
project(GeneticAlgorithm VERSION 1.0.0 LANGUAGES CXX)

find_package(Threads REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find source files
file(GLOB_RECURSE CROSSOVER_SOURCES "crossover/*.cc")
file(GLOB_RECURSE MUTATION_SOURCES "mutation/*.cc")
file(GLOB_RECURSE SELECTION_SOURCES "selection-operator/*.cc")
file(GLOB_RECURSE FITNESS_SOURCES "simple-GA-Test/fitness-function.cc")
set(CORE_SOURCES
    src/genetic_algorithm.cpp
    src/c_api.cpp
    src/algorithms/moea/nsga2.cpp
    src/aco/ant_colony.cpp
    src/aco/continuous_ant_colony.cpp
    src/fuzzy/fuzzy_c_means.cpp
    src/gsa/gravitational_search.cpp
    src/hybrid/metaheuristic_pipeline.cpp
    src/pso/particle_swarm.cpp
)

# Build reusable library (framework)
add_library(genetic_algorithm STATIC
    ${CORE_SOURCES}
    ${CROSSOVER_SOURCES}
    ${MUTATION_SOURCES}
    ${SELECTION_SOURCES}
)
target_include_directories(genetic_algorithm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(genetic_algorithm PROPERTIES OUTPUT_NAME "genetic_algorithm")
target_link_libraries(genetic_algorithm PUBLIC Threads::Threads)

# Warnings: keep them compiler-appropriate (avoid injecting GCC/Clang flags into MSVC).
if(MSVC)
    target_compile_options(genetic_algorithm PRIVATE /W4)
else()
    target_compile_options(genetic_algorithm PRIVATE -Wall -Wextra)
endif()

if(NOT SKBUILD)
    enable_testing()

    # Main executable
    add_executable(simple-ga-test
        simple-ga-test.cc
        ${FITNESS_SOURCES}
    )
    target_link_libraries(simple-ga-test PRIVATE genetic_algorithm)

    # Set target properties
    set_target_properties(simple-ga-test PROPERTIES
        OUTPUT_NAME "simple_ga_test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# Add compiler-specific flags (use target_compile_options for ALL targets via INTERFACE)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(genetic_algorithm PRIVATE -fPIC)
    if(NOT SKBUILD)
        target_compile_options(simple-ga-test PRIVATE -fPIC)
    endif()
endif()

# Install targets
#
# When building a Python wheel via scikit-build (SKBUILD=ON), we only want to
# install the Python extension module. Installing the C++ library/executables
# into the wheel is usually unintended and can bloat the wheel.
if(NOT SKBUILD)
    install(TARGETS genetic_algorithm simple-ga-test
        EXPORT GeneticAlgorithmTargets
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
    install(EXPORT GeneticAlgorithmTargets
        NAMESPACE Genetic::
        DESTINATION lib/cmake/GeneticAlgorithm
    )
endif()

if(NOT SKBUILD)
    # Examples
    add_executable(ga-minimal examples/minimal.cpp)
    target_link_libraries(ga-minimal PRIVATE genetic_algorithm)
    set_target_properties(ga-minimal PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
    )

    add_executable(ga-nsga2-minimal examples/nsga2_minimal.cpp)
    target_link_libraries(ga-nsga2-minimal PRIVATE genetic_algorithm)
    set_target_properties(ga-nsga2-minimal PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
    )

    add_executable(ga-optimizer-minimal examples/optimizer_minimal.cpp)
    target_link_libraries(ga-optimizer-minimal PRIVATE genetic_algorithm)
    set_target_properties(ga-optimizer-minimal PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
    )

    add_executable(ga-metaheuristics-minimal examples/metaheuristics_minimal.cpp)
    target_link_libraries(ga-metaheuristics-minimal PRIVATE genetic_algorithm)
    set_target_properties(ga-metaheuristics-minimal PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
    )

    # Tests (lightweight sanity checks)
    add_executable(operators-sanity tests/operators_sanity.cc)
    target_link_libraries(operators-sanity PRIVATE genetic_algorithm)
    set_target_properties(operators-sanity PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    add_executable(nsga2-sanity tests/nsga2_sanity.cc)
    target_link_libraries(nsga2-sanity PRIVATE genetic_algorithm)
    set_target_properties(nsga2-sanity PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    add_executable(nsga3-sanity tests/nsga3_sanity.cc)
    target_link_libraries(nsga3-sanity PRIVATE genetic_algorithm)
    set_target_properties(nsga3-sanity PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    add_executable(c-api-sanity tests/c_api_sanity.cc)
    target_link_libraries(c-api-sanity PRIVATE genetic_algorithm)
    set_target_properties(c-api-sanity PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    add_executable(features-foundation-sanity tests/features_foundation_sanity.cc)
    target_link_libraries(features-foundation-sanity PRIVATE genetic_algorithm)
    set_target_properties(features-foundation-sanity PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    add_executable(process-distributed-sanity tests/process_distributed_sanity.cc)
    target_link_libraries(process-distributed-sanity PRIVATE genetic_algorithm)
    set_target_properties(process-distributed-sanity PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    add_executable(advanced-features-sanity tests/advanced_features_sanity.cc)
    target_link_libraries(advanced-features-sanity PRIVATE genetic_algorithm)
    set_target_properties(advanced-features-sanity PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    add_executable(representations-core-sanity tests/representations_core_sanity.cc)
    target_link_libraries(representations-core-sanity PRIVATE genetic_algorithm)
    set_target_properties(representations-core-sanity PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    add_executable(metaheuristics-sanity tests/metaheuristics_sanity.cc)
    target_link_libraries(metaheuristics-sanity PRIVATE genetic_algorithm)
    set_target_properties(metaheuristics-sanity PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    add_test(NAME operators-sanity COMMAND operators-sanity)
    add_test(NAME nsga2-sanity COMMAND nsga2-sanity)
    add_test(NAME nsga3-sanity COMMAND nsga3-sanity)
    add_test(NAME c-api-sanity COMMAND c-api-sanity)
    add_test(NAME features-foundation-sanity COMMAND features-foundation-sanity)
    add_test(NAME process-distributed-sanity COMMAND process-distributed-sanity)
    add_test(NAME advanced-features-sanity COMMAND advanced-features-sanity)
    add_test(NAME representations-core-sanity COMMAND representations-core-sanity)
    add_test(NAME metaheuristics-sanity COMMAND metaheuristics-sanity)

    # Benchmark suite
    add_executable(ga-benchmark
        benchmark/benchmark_main.cc
        benchmark/ga_benchmark.cc
    )
    target_link_libraries(ga-benchmark PRIVATE genetic_algorithm)
    set_target_properties(ga-benchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    add_executable(metaheuristics-benchmark benchmark/metaheuristics_benchmark.cc)
    target_link_libraries(metaheuristics-benchmark PRIVATE genetic_algorithm)
    set_target_properties(metaheuristics-benchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# ---------------------------------------------------------------- Python bindings
# When building a Python wheel (SKBUILD=ON), Python bindings are required.
if(SKBUILD)
    # Manylinux Python distributions used by cibuildwheel do not ship libpython
    # (Development.Embed / Python3_LIBRARIES). For extension modules we only
    # need headers + module build info.
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
else()
    find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
endif()

# Try to locate pybind11 via its CMake config (installed via pip or system).
# scikit-build-core already adds site-packages to CMAKE_PREFIX_PATH, but this
# extra hint helps with manual CMake builds.
if(Python3_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _pybind11_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_pybind11_cmake_dir)
        list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmake_dir}")
    endif()
endif()

if(SKBUILD)
    find_package(pybind11 CONFIG REQUIRED)
else()
    find_package(pybind11 CONFIG QUIET)
endif()

if(Python3_FOUND AND pybind11_FOUND)
    # Build the `_core` Python extension module
    pybind11_add_module(ga_python_module python/ga_bindings.cpp)
    target_sources(ga_python_module PRIVATE benchmark/ga_benchmark.cc)
    target_link_libraries(ga_python_module PRIVATE genetic_algorithm)
    target_include_directories(ga_python_module PRIVATE ${CMAKE_SOURCE_DIR}/include)

    if(MSVC)
        target_compile_options(ga_python_module PRIVATE /W4)
    else()
        target_compile_options(ga_python_module PRIVATE -Wall -Wextra)
    endif()

    set_target_properties(ga_python_module PROPERTIES
        OUTPUT_NAME "_core"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    )
    if(SKBUILD)
        install(TARGETS ga_python_module
            LIBRARY DESTINATION genetic_algorithm_lib
            RUNTIME DESTINATION genetic_algorithm_lib
        )
    else()
        install(TARGETS ga_python_module
            LIBRARY DESTINATION lib/python/genetic_algorithm_lib
            RUNTIME DESTINATION lib/python/genetic_algorithm_lib
        )
        add_test(
            NAME python-bindings-sanity
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/python/bindings_sanity.py"
        )
        set_tests_properties(python-bindings-sanity PROPERTIES
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            ENVIRONMENT "GA_PYTHON_MODULE_DIR=${CMAKE_BINARY_DIR}/python"
        )
    endif()
    message(STATUS "Python bindings: ENABLED (pybind11 ${pybind11_VERSION})")
else()
    message(STATUS "Python bindings: DISABLED (requires Python3 + pybind11)")
endif()

# Add custom targets
if(NOT SKBUILD)
    add_custom_target(run
        COMMAND ${CMAKE_BINARY_DIR}/bin/simple_ga_test
        DEPENDS simple-ga-test
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        COMMENT "Running genetic algorithm test"
    )

    add_custom_target(benchmark
        COMMAND ${CMAKE_BINARY_DIR}/bin/ga-benchmark
        DEPENDS ga-benchmark
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        COMMENT "Running genetic algorithm benchmarks"
    )

    add_custom_target(clean-results
        COMMAND ${CMAKE_COMMAND} -E remove -f *.txt
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        COMMENT "Cleaning result files"
    )
endif()

# Print configuration summary
message(STATUS "=== Genetic Algorithm Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Output directory: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "=============================================")
