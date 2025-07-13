# Crossover Operators Library

This directory contains a comprehensive collection of crossover operators for genetic algorithms, each implemented in separate files for better modularity and maintainability.

## Directory Structure

```
crossover/
├── base_crossover.h                    # Base class definition
├── base_crossover.cc                   # Base class implementation
├── all_crossover_operators.h           # Main header including all operators
├── all_crossover_operators.cc          # Factory functions and utilities
└── individual operator files...
```

## Available Crossover Operators

### Bit-String/Vector Encodings
- **OnePointCrossover** (`one_point_crossover.h/cc`) - Single crossover point
- **TwoPointCrossover** (`two_point_crossover.h/cc`) - Two crossover points
- **MultiPointCrossover** (`multi_point_crossover.h/cc`) - Multiple crossover points
- **UniformCrossover** (`uniform_crossover.h/cc`) - Probabilistic gene exchange
- **UniformKVectorCrossover** (`uniform_k_vector_crossover.h/cc`) - Multi-parent uniform crossover

### Real-Valued Encodings
- **BlendCrossover** (`blend_crossover.h/cc`) - BLX-α crossover
- **SimulatedBinaryCrossover** (`simulated_binary_crossover.h/cc`) - SBX crossover
- **LineRecombination** (`line_recombination.h/cc`) - Linear recombination
- **IntermediateRecombination** (`intermediate_recombination.h/cc`) - Arithmetic recombination

### Permutation Encodings
- **CutAndCrossfillCrossover** (`cut_and_crossfill_crossover.h/cc`) - Cut and fill approach
- **PartiallyMappedCrossover** (`partially_mapped_crossover.h/cc`) - PMX crossover
- **EdgeCrossover** (`edge_crossover.h/cc`) - Edge-based crossover
- **OrderCrossover** (`order_crossover.h/cc`) - OX crossover
- **CycleCrossover** (`cycle_crossover.h/cc`) - CX crossover

### Tree-Based Encodings
- **SubtreeCrossover** (`subtree_crossover.h/cc`) - Genetic programming crossover

### Specialized Encodings
- **DiploidRecombination** (`diploid_recombination.h/cc`) - Diploid chromosome crossover
- **DifferentialEvolutionCrossover** (`differential_evolution_crossover.h/cc`) - DE-style crossover

## Usage

### Basic Usage

```cpp
#include "all_crossover_operators.h"

// Create a crossover operator
auto crossover_op = CrossoverOperators::create("OnePoint");

// Use with binary encoding
BitString parent1 = {true, false, true, false};
BitString parent2 = {false, true, false, true};
auto [child1, child2] = crossover_op->crossover(parent1, parent2);

// Use with real-valued encoding
RealVector parent1_real = {1.0, 2.0, 3.0};
RealVector parent2_real = {4.0, 5.0, 6.0};
auto [child1_real, child2_real] = crossover_op->crossover(parent1_real, parent2_real);
```

### Factory Function

```cpp
// Create different types of crossover operators
auto one_point = CrossoverOperators::create("OnePoint");
auto two_point = CrossoverOperators::create("TwoPoint");
auto blend = CrossoverOperators::create("Blend");
auto pmx = CrossoverOperators::create("PMX");
```

### Available Operator Names

The factory function accepts the following names (case-insensitive):

**Bit-String/Vector:**
- "OnePoint", "one_point", "one-point"
- "TwoPoint", "two_point", "two-point"
- "MultiPoint", "multi_point", "multi-point"
- "Uniform"
- "UniformKVector", "uniform_k_vector"

**Real-Valued:**
- "Blend", "BLX", "blx-alpha"
- "SimulatedBinary", "SBX", "simulated_binary"
- "LineRecombination", "line_recombination"
- "Intermediate", "intermediate_recombination"

**Permutation:**
- "CutAndCrossfill", "cut_and_crossfill"
- "PartiallyMapped", "PMX", "pmx"
- "Edge", "edge_crossover"
- "Order", "OX", "ox", "order_crossover"
- "Cycle", "CX", "cx", "cycle_crossover"

**Specialized:**
- "Subtree", "subtree_crossover"
- "Diploid", "diploid_recombination"
- "DifferentialEvolution", "DE", "de"

## Algorithm-Specific Features

### Statistics Tracking
All crossover operators inherit statistics tracking:
```cpp
auto op = CrossoverOperators::create("OnePoint");
// ... perform crossovers ...
std::cout << "Operations: " << op->getOperationCount() << std::endl;
std::cout << "Errors: " << op->getErrorCount() << std::endl;
std::cout << "Error Rate: " << op->getErrorRate() << std::endl;
```

### Parameter Configuration
Many operators accept parameters during construction:
```cpp
// Blend crossover with alpha = 0.3
auto blend = std::make_unique<BlendCrossover>(0.3);

// Multi-point crossover with 5 points
auto multi = std::make_unique<MultiPointCrossover>(5);

// SBX with distribution index = 5.0
auto sbx = std::make_unique<SimulatedBinaryCrossover>(5.0);
```

## Building

Add all crossover files to your build system. For MBDyn's autotools system, the files are already included in `Makefile.inc`.

## Testing

Run the test program to verify all operators work correctly:
```bash
./test_crossover_operators
```

## Adding New Crossover Operators

1. Create header file: `new_crossover.h`
2. Create implementation file: `new_crossover.cc`
3. Inherit from `CrossoverOperator` base class
4. Add to `all_crossover_operators.h` includes
5. Add to factory function in `all_crossover_operators.cc`
6. Update this README

## Type Definitions

```cpp
using BitString = std::vector<bool>;
using RealVector = std::vector<double>;
using IntVector = std::vector<int>;
using Permutation = std::vector<int>;
```

## Error Handling

All crossover operators throw `std::invalid_argument` for:
- Mismatched parent sizes
- Invalid parameters
- Null pointers (for tree operations)

Operations are tracked and error counts are maintained for debugging purposes.
