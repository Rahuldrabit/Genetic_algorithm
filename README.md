# Genetic Algorithm Framework (C++)

A reusable C++ genetic algorithm framework you can embed in any application. It exposes a small, modern C++ API and ships with a rich set of crossover, mutation, and selection operators.

## 🚀 Features

- **Multi-Representation Support**: Binary, Real-valued, Integer, and Permutation representations
- **Comprehensive Operators**: 35+ crossover, mutation, and selection operators
- **Benchmark Functions**: Rastrigin, Ackley, Schwefel, Rosenbrock, and Sphere optimization problems
- **Modern Build System**: CMake-based build configuration
- **Cross-Platform**: Works on Linux, macOS, and Windows
- **Multi-Language Support**: C++ (primary), Python bindings, and C-compatible interfaces
- **Performance Benchmarks**: Comprehensive benchmark suite for operators and functions
- **Production-Ready**: Modern C++17 with smart pointers and RAII

## Technical Checklist and Roadmap

- Complete feature checklist (DEAP-level + beyond): [FEATURE_CHECKLIST.md](FEATURE_CHECKLIST.md)
- Architecture overview and usage guidance: [ARCHITECTURE.md](ARCHITECTURE.md)

## 📁 Project Structure

```
Genetic_algorithm/
├── CMakeLists.txt              # Main CMake configuration
├── README.md                   # This file
├── include/ga/                 # Public framework headers (installable)
│   ├── config.hpp              # Config, Bounds, Result, Fitness alias
│   └── genetic_algorithm.hpp   # GeneticAlgorithm class and factories
├── src/
│   └── genetic_algorithm.cpp   # Core GA engine implementation
├── examples/
│   └── minimal.cpp             # Tiny example app using the framework
├── simple-ga-test.cc           # Legacy interactive demo (still works)
├── crossover/                  # Crossover operators
│   ├── base_crossover.h/cc     # Base crossover interface
│   ├── one_point_crossover.h/cc
│   ├── two_point_crossover.h/cc
│   ├── uniform_crossover.h/cc
│   ├── blend_crossover.h/cc
│   ├── simulated_binary_crossover.h/cc
│   ├── order_crossover.h/cc
│   ├── partially_mapped_crossover.h/cc
│   ├── cycle_crossover.h/cc
│   └── ... (15+ more operators)
├── mutation/                   # Mutation operators
│   ├── base_mutation.h/cc      # Base mutation interface
│   ├── bit_flip_mutation.h/cc
│   ├── gaussian_mutation.h/cc
│   ├── uniform_mutation.h/cc
│   ├── swap_mutation.h/cc
│   └── ... (10+ more operators)
├── selection-operator/         # Selection methods
│   ├── base_selection.h/cc     # Base selection interface
│   ├── tournament_selection.h/cc
│   ├── roulette_wheel_selection.h/cc
│   ├── rank_selection.h/cc
│   └── ... (5+ more operators)
├── benchmark/                  # Benchmark suite (NEW!)
│   ├── ga_benchmark.h/cc       # Comprehensive benchmarks
│   └── benchmark_main.cc       # Benchmark executable
└── simple-GA-Test/             # Test suite and fitness functions
    ├── fitness-function.h      # Fitness function declarations
    ├── fitness-fuction.cc      # Fitness function implementations
    └── README.md              # Detailed test documentation
```

## 🛠️ Building with CMake

### Prerequisites

- **CMake** (version 3.16 or higher)
- **C++17 compatible compiler**:
  - GCC 7+ (Linux/macOS)
  - Clang 5+ (Linux/macOS)
  - MSVC 2017+ (Windows)

### Quick Start

#### Using Build Script (Recommended)

```bash
# Clone or navigate to the project directory
cd test-ga

# Build and run in one command
./build.sh --run

# Or build only
./build.sh
```

#### Using CMake Directly

```bash
cd Genetic_algorithm

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build .

# Run the legacy demo
./bin/simple_ga_test
```

### Advanced Build Options

```bash
# Debug build with verbose output
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --verbose

# Release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)

# Install to system (optional)
sudo cmake --build . --target install
```

### Build Script

The project includes a convenient build script (`build.sh`) that automates the build process:

```bash
# Basic build
./build.sh

# Build with options
./build.sh --debug --run --verbose

# Clean build
./build.sh --clean

# Install to system
./build.sh --install
```

### CMake Targets

- `genetic_algorithm`: Static library (the framework)
- `simple-ga-test`: Interactive demo executable
- `run`: Build and run the GA test
- `clean-results`: Remove output files
- `install`: Install to system

```bash
# Use custom targets
cmake --build . --target run
cmake --build . --target clean-results
```

## 🎯 Using the framework in your code

The public API is in `include/ga`. Example:

```cpp
#include <ga/genetic_algorithm.hpp>
#include <cmath>

static double rastrigin(const std::vector<double>& x) {
  const double A = 10.0;
  double sum = A * x.size();
  for (double xi : x) sum += xi*xi - A*std::cos(2*M_PI*xi);
  // convert minimization to maximization fitness
  return 1000.0 / (1.0 + sum);
}

int main() {
  ga::Config cfg;
  cfg.populationSize = 60;
  cfg.generations = 100;
  cfg.dimension = 10;
  cfg.bounds = {-5.12, 5.12};

  ga::GeneticAlgorithm alg(cfg);
  ga::Result res = alg.run(rastrigin);
}
```

You can also compile and run the ready-made example:

```bash
cmake --build build -j
./build/examples/minimal
```

NSGA-II minimal example:

```bash
cmake --build build -j
./build/examples/ga-nsga2-minimal
```

High-level optimizer API example:

```bash
cmake --build build -j
./build/examples/ga-optimizer-minimal
```

NSGA-III sanity test:

```bash
cmake --build build --target nsga3-sanity
./build/tests/nsga3-sanity
```

To customize operators:

```cpp
#include <ga/genetic_algorithm.hpp>

auto alg = ga::GeneticAlgorithm(cfg);
alg.setCrossoverOperator(ga::makeTwoPointCrossover());
alg.setMutationOperator(ga::makeUniformMutation());
```

### C API (Baseline Wrapper)

The project includes a C-compatible API in `include/ga/c_api.h`.

Key C API additions:
- `ga_validate_config(...)` for pre-run argument checks
- `ga_history_length(...)`, `ga_best_history(...)`, `ga_avg_history(...)` for convergence history export

```c
#include <ga/c_api.h>

static double sphere_fitness(const double* genes, int length, void* user_data) {
  (void)user_data;
  double sum = 0.0;
  for (int i = 0; i < length; ++i) {
    sum += genes[i] * genes[i];
  }
  return 1000.0 / (1.0 + sum);
}

int main(void) {
  ga_config_c cfg = {60, 100, 10, 0.8, 0.1, -5.12, 5.12, 0.05, 42};
  if (ga_validate_config(&cfg) != GA_STATUS_OK) {
    return 1;
  }

  ga_handle* h = ga_create(&cfg);
  if (!h) {
    return 1;
  }

  if (ga_run(h, sphere_fitness, 0) != GA_STATUS_OK) {
    ga_destroy(h);
    return 1;
  }

  double best = ga_best_fitness(h);

  int n = ga_history_length(h);
  double history[1024];
  if (n > 0 && n <= 1024) {
    (void)ga_best_history(h, history, n);
  }

  ga_destroy(h);
  (void)best;
  return 0;
}
```

### NSGA-II Utilities (C++ Core)

NSGA-II helper APIs are available in `include/ga/algorithms/moea/nsga2.hpp`:
- non-dominated sorting
- crowding distance
- callback-based generation loop (`run(...)`)

NSGA-III helper APIs are available in `include/ga/moea/nsga3.hpp`:
- Das-Dennis reference point generation
- reference-point environmental selection niching with intercept-based normalization

High-level optimizer methods in `include/ga/api/optimizer.hpp` now include:
- `optimizeMultiObjective(...)` (NSGA-II)
- `optimizeMultiObjectiveNsga3(...)`

Distributed evaluation backends in `include/ga/evaluation/distributed_executor.hpp`:
- `LocalDistributedExecutor` (threaded local backend)
- `ProcessDistributedExecutor` (true multi-process backend on POSIX)

Python bindings (`python/ga_bindings.cpp`) also expose:
- NSGA-III objective-space utilities (`Nsga3`, `nsga3_reference_points`, `nsga3_environmental_select_indices`)
- checkpoint JSON API (`CheckpointState`, `checkpoint_save_json`, `checkpoint_load_json`)

### Interactive Mode (Recommended)

```bash
./bin/simple_ga_test
```

The program will guide you through:
1. **Representation selection** (binary, real_valued, integer, permutation)
2. **Operator validation** against chosen representation
3. **Automatic configuration** of compatible operators

### Command Line Testing

#### Real-Valued Optimization
```bash
echo -e "real_valued\nblend\ngaussian\ntournament" | ./bin/simple_ga_test
```

#### Binary Optimization
```bash
echo -e "binary\nuniform\nbit_flip\ntournament" | ./bin/simple_ga_test
```

#### Integer Optimization
```bash
echo -e "integer\narithmetic\nrandom_resetting\ntournament" | ./bin/simple_ga_test
```

#### Permutation Problems
```bash
echo -e "permutation\norder_crossover\nswap\ntournament" | ./bin/simple_ga_test
```

#### C API Sanity Test
```bash
cmake --build build --target c-api-sanity
./build/tests/c-api-sanity
```

#### Feature Foundation Sanity Test
```bash
cmake --build build --target features-foundation-sanity
./build/tests/features-foundation-sanity
```

#### Process Distributed Backend Sanity Test
```bash
cmake --build build --target process-distributed-sanity
./build/tests/process-distributed-sanity
```

## 🔧 Configuration

The framework uses `ga::Config`:

```cpp
struct Bounds { double lower, upper; };
struct Config {
  int populationSize = 50;
  int generations = 100;
  int dimension = 10;
  double crossoverRate = 0.8;
  double mutationRate = 0.1;
  Bounds bounds{-5.12, 5.12};
  double eliteRatio = 0.05; // 5% elites
  unsigned seed = 0;        // 0 -> random
};
```

## 📊 Supported Representations & Operators

### Binary Representation
- **Crossovers**: One-point, Two-point, Uniform
- **Mutations**: Bit-flip
- **Use Cases**: Feature selection, binary optimization

### Real-Valued Representation
- **Crossovers**: Arithmetic, Blend (BLX-α), SBX, One-point, Two-point, Uniform
- **Mutations**: Gaussian, Uniform
- **Use Cases**: Continuous function optimization, parameter tuning

### Integer Representation
- **Crossovers**: One-point, Two-point, Uniform, Arithmetic
- **Mutations**: Random resetting, Creep
- **Use Cases**: Discrete optimization, scheduling problems

### Permutation Representation
- **Crossovers**: Order crossover (OX), Partially mapped crossover (PMX), Cycle crossover
- **Mutations**: Swap, Insert, Scramble, Inversion
- **Use Cases**: Traveling salesman problem, job scheduling

## 🧪 Benchmark Functions

The framework includes 5 standard optimization test functions:

1. **Sphere Function**: Simple unimodal function (baseline)
2. **Rastrigin Function**: Highly multimodal with many local optima
3. **Ackley Function**: One global minimum with many local minima
4. **Schwefel Function**: Deceptive function with global optimum far from local optima
5. **Rosenbrock Function**: Narrow valley, challenging for optimization

## 🔬 Running Benchmarks

The framework includes a comprehensive benchmark suite that tests:
- **Operator Performance**: Speed of crossover, mutation, and selection operators
- **Function Optimization**: Convergence quality on test functions
- **Scalability**: Performance vs. population size and problem dimension

### Quick Start

```bash
# Build the benchmark executable
cmake --build build

# Run all benchmarks
./build/bin/ga-benchmark --all

# Run specific benchmark categories
./build/bin/ga-benchmark --operators      # Test operator performance
./build/bin/ga-benchmark --functions      # Test function optimization
./build/bin/ga-benchmark --scalability    # Test scalability

# Customize benchmark iterations
./build/bin/ga-benchmark --operators --iterations 1000

# Export results to CSV
./build/bin/ga-benchmark --all --csv

# Show help
./build/bin/ga-benchmark --help
```

### Benchmark Results

**Operator Performance (typical results on modern CPU):**
| Operator Category | Representative | Throughput |
|-------------------|----------------|------------|
| Binary Crossover | TwoPointCrossover | 2M ops/sec |
| Real Crossover | BlendCrossover (BLX-α) | 5M ops/sec |
| Permutation Crossover | OrderCrossover (OX) | 869K ops/sec |
| Binary Mutation | BitFlipMutation | 1.1M ops/sec |
| Real Mutation | GaussianMutation | 6.6M ops/sec |
| Permutation Mutation | SwapMutation | 20M ops/sec |
| Selection | TournamentSelection | 181K ops/sec |

**Function Optimization (convergence times):**
| Function | Generations | Time (ms) | Best Fitness |
|----------|-------------|-----------|--------------|
| Sphere | 100 | ~1 | >500 |
| Rastrigin | 200 | ~5 | >60 |
| Ackley | 150 | ~4 | >60 |
| Schwefel | 200 | ~7 | Variable |
| Rosenbrock | 300 | ~8 | >200 |

*Results will vary based on hardware, problem configuration, and random seed.*

### Understanding Benchmark Output

The benchmark tool generates:
- **Console output**: Real-time progress and summary statistics
- **benchmark_results.txt**: Detailed results with all metrics
- **benchmark_results.csv**: Machine-readable format (with `--csv` flag)

## 🏗️ Architecture & Efficiency

For a detailed analysis of the framework's architecture, efficiency, and usability across C++, Python, and C, see [ARCHITECTURE.md](ARCHITECTURE.md).

**Key Highlights:**
- ⚡ **Performance**: Native C++17 with zero-overhead abstractions
- 🔧 **Extensible**: Easy to add custom operators and fitness functions
- 🌐 **Multi-language**: C++ core with Python bindings
- 📊 **Validated**: Comprehensive benchmark suite included
- 🧪 **Tested**: Multiple test programs and sanity checks

## 🔍 Development

### Adding New Operators

1. Create header and implementation files in the appropriate directory
2. Inherit from the base operator class
3. Implement required virtual methods
4. Optionally expose convenience factories alongside `ga::make*` helpers

### Adding New Fitness Functions

1. Add declaration to `simple-GA-Test/fitness-function.h`
2. Implement in `simple-GA-Test/fitness-fuction.cc`
3. Add to the `GAConfig::FunctionType` enum
4. Update the fitness function selection logic

### Building for Development

```bash
# Debug build with symbols
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .

# Run with debug output
./bin/simple_ga_test
```

## 📝 Output

The program generates:
- **Console output**: Progress information and final results
- **ga_results.txt**: Detailed results including:
  - Best fitness values per generation
  - Average fitness values
  - Best individual's chromosome
  - Optimization statistics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Please check individual files for license information.

## 🆘 Troubleshooting

### Common Issues

**CMake not found:**
```bash
# Ubuntu/Debian
sudo apt install cmake

# macOS
brew install cmake

# Windows
# Download from https://cmake.org/download/
```

**Compiler not found:**
```bash
# Ubuntu/Debian
sudo apt install build-essential

# macOS
xcode-select --install
```

**Build errors:**
```bash
# Clean and rebuild
rm -rf build
mkdir build && cd build
cmake ..
cmake --build .
```

### Getting Help

- Check the detailed documentation in `simple-GA-Test/README.md`
- Review the CMake configuration in `CMakeLists.txt`
- Examine the source code for implementation details 