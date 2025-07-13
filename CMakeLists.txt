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

# Find source files
file(GLOB_RECURSE CROSSOVER_SOURCES "crossover/*.cc")
file(GLOB_RECURSE MUTATION_SOURCES "mutation/*.cc")
file(GLOB_RECURSE SELECTION_SOURCES "selection-operator/*.cc")
file(GLOB_RECURSE FITNESS_SOURCES "simple-GA-Test/fitness-fuction.cc")

# Main executable
add_executable(simple-ga-test
    simple-ga-test.cc
    ${FITNESS_SOURCES}
    ${CROSSOVER_SOURCES}
    ${MUTATION_SOURCES}
    ${SELECTION_SOURCES}
)

# Set target properties
set_target_properties(simple-ga-test PROPERTIES
    OUTPUT_NAME "simple_ga_test"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Add compiler-specific flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(simple-ga-test PRIVATE -fPIC)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(simple-ga-test PRIVATE -fPIC)
endif()

# Install target
install(TARGETS simple-ga-test
    RUNTIME DESTINATION bin
)

# Add custom targets
add_custom_target(run
    COMMAND ${CMAKE_BINARY_DIR}/bin/simple_ga_test
    DEPENDS simple-ga-test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "Running genetic algorithm test"
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