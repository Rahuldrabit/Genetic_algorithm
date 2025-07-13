# Genetic Algorithm Framework

A comprehensive C++ implementation of genetic algorithms with support for multiple representation types and optimization operators.

## 🚀 Features

- **Multi-Representation Support**: Binary, Real-valued, Integer, and Permutation representations
- **Comprehensive Operators**: 20+ crossover and mutation operators
- **Benchmark Functions**: Rastrigin, Ackley, and Schwefel optimization problems
- **Modern Build System**: CMake-based build configuration
- **Cross-Platform**: Works on Linux, macOS, and Windows

## 📁 Project Structure

```
test-ga/
├── CMakeLists.txt              # Main CMake configuration
├── README.md                   # This file
├── simple-ga-test.cc           # Main GA implementation
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
# Clone or navigate to the project directory
cd test-ga

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build .

# Run the genetic algorithm
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

- `simple-ga-test`: Main executable
- `run`: Build and run the GA test
- `clean-results`: Remove output files
- `install`: Install to system

```bash
# Use custom targets
cmake --build . --target run
cmake --build . --target clean-results
```

## 🎯 Usage Examples

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

The genetic algorithm can be configured through the `GAConfig` structure:

```cpp
struct GAConfig {
    int populationSize = 50;        // Population size
    int generations = 50;           // Number of generations
    int chromosomeLength = 10;      // Number of variables
    double mutationRate = 0.01;     // Mutation probability
    double crossoverRate = 0.8;     // Crossover probability
    double eliteRatio = 0.1;        // Elite preservation ratio
    
    // Function bounds
    double lowerBound = -5.12;
    double upperBound = 5.12;
    
    // Function and representation selection
    enum FunctionType { RASTRIGIN, ACKLEY, SCHWEFEL } function = RASTRIGIN;
    enum RepresentationType { BINARY, REAL_VALUED, INTEGER, PERMUTATION } representation = REAL_VALUED;
    
    // Operator selections (auto-validated)
    std::string crossoverType = "one_point";
    std::string mutationType = "gaussian";
    std::string selectionType = "tournament";
    
    // Output settings
    bool verbose = true;
    std::string outputFile = "ga_results.txt";
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

1. **Rastrigin Function**: Highly multimodal with many local optima
2. **Ackley Function**: One global minimum with many local minima
3. **Schwefel Function**: Deceptive function with global optimum far from local optima

## 🔍 Development

### Adding New Operators

1. Create header and implementation files in the appropriate directory
2. Inherit from the base operator class
3. Implement required virtual methods
4. Add to the factory functions in `simple-ga-test.cc`

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