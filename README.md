# Genetic Algorithm Framework (C++)

A reusable C++ genetic algorithm framework you can embed in any application. It exposes a small, modern C++ API and ships with a rich set of crossover, mutation, and selection operators, plus a full **Genetic Programming (GP)** engine.

## 🚀 Features

- **Multi-Representation Support**: Binary, Real-valued, Integer, Permutation, **Set, Dictionary, Graph, and Tree** representations
- **Polymorphic Individual Hierarchy**: `VectorIndividual`, `SetIndividual`, `DictIndividual`, `TreeIndividual`, `GraphIndividual` — all deriving from a common `ga::Individual` base class
- **Genetic Programming (GP)**: Prefix expression-tree representation for symbolic regression and program evolution
- **Strongly-Typed & Loosely-Typed GP**: Enforce type safety (`GPType::Real/Bool/Int`) or allow free-form combination
- **Comprehensive Operators**: 35+ crossover, mutation, and selection operators, including `SubtreeCrossover` and `PointMutation` for GP
- **Benchmark Functions**: Rastrigin, Ackley, Schwefel, Rosenbrock, and Sphere optimization problems
- **Modern Build System**: CMake-based build configuration
- **Cross-Platform**: Works on Linux, macOS, and Windows
- **Multi-Language Support**: C++ (primary), Python bindings, and C-compatible interfaces
- **Performance Benchmarks**: Comprehensive benchmark suite for operators and functions
- **Production-Ready**: Modern C++17 with smart pointers and RAII

## 📁 Project Structure

```
Genetic_algorithm/
├── CMakeLists.txt              # Main CMake configuration
├── README.md                   # This file
├── include/ga/                 # Public framework headers (installable)
│   ├── config.hpp              # Config, Bounds, Result, Fitness alias
│   ├── genetic_algorithm.hpp   # GeneticAlgorithm class and factories
│   ├── individual.hpp          # Polymorphic Individual class hierarchy
│   └── gp/                     # Genetic Programming sub-system
│       ├── expression_tree.hpp # ExprNode, PrimitiveSet, TypeChecker
│       └── gp_engine.hpp       # GPConfig, GPResult, GPEngine
├── src/
│   ├── genetic_algorithm.cpp   # Core GA engine implementation
│   ├── individual.cpp          # TreeIndividual method implementations
│   └── gp/
│       ├── expression_tree.cpp # GP tree evaluation
│       └── gp_engine.cpp       # GP algorithm (init, crossover, mutation, loop)
├── examples/
│   └── minimal.cpp             # Tiny example app using the framework
├── simple-ga-test.cc           # Legacy interactive demo (still works)
├── crossover/                  # Crossover operators
│   ├── base_crossover.h/cc     # Base crossover interface + TreeNode
│   ├── subtree_crossover.h/cc  # GP subtree crossover
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
│   ├── point_mutation.h/cc     # GP point mutation (node replacement)
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
├── benchmark/                  # Benchmark suite
│   ├── ga_benchmark.h/cc       # Comprehensive benchmarks
│   └── benchmark_main.cc       # Benchmark executable
├── tests/
│   ├── operators_sanity.cc     # Sanity tests for all 35+ operators
│   └── gp_sanity.cc            # Sanity tests for GP features
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
- `operators-sanity`: Sanity test for all 35+ classical operators
- `gp-sanity`: Sanity test for GP features (expression trees, typed GP, individual hierarchy)
- `run`: Build and run the GA test
- `clean-results`: Remove output files
- `install`: Install to system

```bash
# Use custom targets
cmake --build . --target run
cmake --build . --target clean-results
```

## 🌳 Genetic Programming (GP)

### Individual Representations

The framework provides a polymorphic `ga::Individual` base class with five concrete types:

| Type | Description | Use-case |
|------|-------------|----------|
| `VectorIndividual` | `std::vector<double>` gene vector | Continuous / array optimisation |
| `SetIndividual` | `std::set<int>` | Subset-selection problems |
| `DictIndividual` | `std::unordered_map<string,double>` | Named-parameter optimisation |
| `TreeIndividual` | GP expression tree | Symbolic regression, program evolution |
| `GraphIndividual` | Adjacency-list graph | Network / architecture search |

```cpp
#include <ga/individual.hpp>

// Real-valued (Numpy-like array)
ga::VectorIndividual vi({1.0, 2.0, 3.0},
    [](const std::vector<double>& g){ return g[0]+g[1]+g[2]; });
std::cout << vi.fitness(); // 6.0

// Set chromosome
ga::SetIndividual si({1,2,3},
    [](const std::set<int>& s){ double r=0; for(int v:s) r+=v; return r; });

// Dict chromosome
ga::DictIndividual::GeneMap gm{{"lr", 0.01}, {"dropout", 0.5}};
ga::DictIndividual di(gm, [](const ga::DictIndividual::GeneMap& m){
    return 1.0 / m.at("lr"); });

// Graph chromosome
ga::GraphIndividual::AdjList adj{{1,2},{2},{}};
ga::GraphIndividual gi(adj,
    [](const ga::GraphIndividual::AdjList& a){ return (double)a.size(); });
```

### Expression Trees

GP programs are represented as prefix expression trees (`ga::gp::ExprNode`).
Build a primitive set, then let the GP engine evolve programs:

```cpp
#include <ga/gp/expression_tree.hpp>
#include <ga/gp/gp_engine.hpp>
#include <cmath>

// Build (+ (* x y) 1)
auto root = std::make_unique<ga::gp::ExprNode>("+", ga::gp::GPType::Real);
auto mul  = std::make_unique<ga::gp::ExprNode>("*", ga::gp::GPType::Real);
mul->children.push_back(std::make_unique<ga::gp::ExprNode>("x", ga::gp::GPType::Real));
mul->children.push_back(std::make_unique<ga::gp::ExprNode>("y", ga::gp::GPType::Real));
root->children.push_back(std::move(mul));
root->children.push_back(std::make_unique<ga::gp::ExprNode>("1", ga::gp::GPType::Real));

std::cout << root->toSExpr(); // (+ (* x y) 1)
std::cout << root->size();    // 5
std::cout << root->depth();   // 2
```

### Primitive Set

```cpp
// Use the built-in arithmetic set (+, -, *, /, sin, cos, terminals)
auto pset = ga::gp::PrimitiveSet::makeArithmetic({"x", "y"});

// Evaluate with variable values
double val = ga::gp::evaluate(*root, pset, {2.0, 3.0}); // (* 2 3) + 1 = 7
```

You can also add custom primitives:

```cpp
pset.addFunction("exp2", ga::gp::GPType::Real, {ga::gp::GPType::Real},
    [](const std::vector<double>& a){ return std::exp2(a[0]); });

pset.addTerminal("PI", ga::gp::GPType::Real,
    [](const std::vector<double>&){ return M_PI; });
```

### Strongly-Typed GP

Set `cfg.stronglyTyped = true` to enforce that every parent–child type edge matches the primitive's declared argument types.  Use `ga::gp::TypeChecker` to validate trees:

```cpp
ga::gp::TypeChecker tc(pset);
bool valid = tc.check(*root, ga::gp::GPType::Real); // true if all types agree
```

### Loosely-Typed GP

Set `cfg.stronglyTyped = false` (default).  Any primitive can be combined freely; `GPType::Any` is the wildcard type.

### Running the GP Engine

```cpp
#include <ga/gp/gp_engine.hpp>
#include <cmath>

auto pset = ga::gp::PrimitiveSet::makeArithmetic({"x"});

ga::gp::GPConfig cfg;
cfg.populationSize = 100;
cfg.generations    = 50;
cfg.maxTreeDepth   = 6;
cfg.mutationRate   = 0.1;
cfg.stronglyTyped  = false;  // loosely-typed (default)
cfg.seed           = 42;

// Fitness: symbolic regression towards x^2
std::vector<double> xs = {1.0, 2.0, 3.0, 4.0, 5.0};
ga::gp::GPEngine engine(cfg, pset);
ga::gp::GPResult  res = engine.run([&](const ga::gp::ExprNode& tree){
    double err = 0.0;
    for (double xv : xs) {
        double d = ga::gp::evaluate(tree, pset, {xv}) - xv*xv;
        err += d*d;
    }
    return 1.0 / (1.0 + std::sqrt(err / xs.size()));
});

std::cout << "Best program : " << res.bestProgram()  << '\n';
std::cout << "Best fitness : " << res.bestFitness     << '\n';
```

### GP Point Mutation

`PointMutation` (in `mutation/point_mutation.h`) replaces individual tree nodes with randomly chosen compatible primitives, leaving the tree's arity structure intact:

```cpp
#include "mutation/point_mutation.h"

PointMutation pm(/*seed=*/42);
pm.mutate(tree, pset, /*per-node probability=*/0.1, /*stronglyTyped=*/false);
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

To customize operators:

```cpp
#include <ga/genetic_algorithm.hpp>

auto alg = ga::GeneticAlgorithm(cfg);
alg.setCrossoverOperator(ga::makeTwoPointCrossover());
alg.setMutationOperator(ga::makeUniformMutation());
```

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