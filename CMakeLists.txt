cmake_minimum_required(VERSION 3.16)
project(GeneticAlgorithm VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

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
    src/individual.cpp
    src/gp/expression_tree.cpp
    src/gp/gp_engine.cpp
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

# Add compiler-specific flags (use target_compile_options for ALL targets via INTERFACE)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(genetic_algorithm PRIVATE -fPIC)
    target_compile_options(simple-ga-test PRIVATE -fPIC)
endif()

# Install target
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

# Examples
add_executable(ga-minimal examples/minimal.cpp)
target_link_libraries(ga-minimal PRIVATE genetic_algorithm)
set_target_properties(ga-minimal PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
)

# Tests (lightweight sanity checks)
add_executable(operators-sanity tests/operators_sanity.cc)
target_link_libraries(operators-sanity PRIVATE genetic_algorithm)
set_target_properties(operators-sanity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)

# GP sanity tests
add_executable(gp-sanity tests/gp_sanity.cc)
target_link_libraries(gp-sanity PRIVATE genetic_algorithm)
set_target_properties(gp-sanity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)

# Benchmark suite
add_executable(ga-benchmark
    benchmark/benchmark_main.cc
    benchmark/ga_benchmark.cc
)
target_link_libraries(ga-benchmark PRIVATE genetic_algorithm)
set_target_properties(ga-benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ---------------------------------------------------------------- Python bindings
# Optional: only built when Python3 and pybind11 are available.
find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(Python3_FOUND)
    # Try to locate pybind11 via its cmake config (installed via pip or system)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _pybind11_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_pybind11_cmake_dir)
        list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmake_dir}")
        find_package(pybind11 QUIET)
    endif()
endif()

if(pybind11_FOUND)
    # Build the 'ga' Python extension module
    pybind11_add_module(ga_python_module python/ga_bindings.cpp)
    target_link_libraries(ga_python_module PRIVATE genetic_algorithm)
    target_include_directories(ga_python_module PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(ga_python_module PROPERTIES
        OUTPUT_NAME "ga"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    )
    message(STATUS "Python bindings: ENABLED (pybind11 ${pybind11_VERSION})")
else()
    message(STATUS "Python bindings: DISABLED (pybind11 not found — install with: pip install pybind11)")
endif()

# Add custom targets
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

# Add test target (if you want to add unit tests later)
# enable_testing()
# add_test(NAME GA_Test COMMAND simple-ga-test)

# Print configuration summary
message(STATUS "=== Genetic Algorithm Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Output directory: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "=============================================")
