# Modularized Genetic Algorithm Module for MBDyn

This module provides a comprehensive, modularized genetic algorithm optimization framework for MBDyn. The codebase has been refactored to separate crossover, mutation, and selection operators into individual files, making the system highly maintainable, extensible, and easy to use.

## Architecture Overview

The genetic algorithm module is organized into three main operator categories:

```
module-GeneticAlgorithm/
├── crossover/              # Crossover operators
├── mutation/               # Mutation operators  
├── selection-operator/     # Selection operators
├── module-GeneticAlgorithm.h   # Main GA framework header
├── module-GeneticAlgorithm.cc  # Main GA framework implementation
└── test_complete_ga.cc     # Comprehensive test/example file
```

## Crossover Operators

### Available Operators
- **OnePointCrossover**: Classic single-point crossover
- **TwoPointCrossover**: Two-point crossover
- **MultiPointCrossover**: Configurable multi-point crossover
- **UniformCrossover**: Uniform crossover with probability mask
- **UniformKVectorCrossover**: Uniform crossover with K-vector selection
- **BlendCrossover**: Blend crossover (BLX-α)
- **SimulatedBinaryCrossover**: Simulated binary crossover (SBX)
- **LineRecombination**: Line recombination for real-valued problems
- **IntermediateRecombination**: Intermediate recombination
- **CutAndCrossfillCrossover**: Cut-and-crossfill for permutation problems
- **PartiallyMappedCrossover**: PMX for permutation problems
- **EdgeCrossover**: Edge crossover for TSP-like problems
- **OrderCrossover**: Order crossover (OX) for permutations
- **CycleCrossover**: Cycle crossover for permutations
- **SubtreeCrossover**: Subtree crossover for tree structures
- **DiploidRecombination**: Diploid recombination
- **DifferentialEvolutionCrossover**: DE/rand/1/bin crossover

### Usage
```cpp
#include "crossover/all_crossover_operators.h"

// Create a crossover operator
auto crossover_op = crossover::CreateCrossoverOperator("simulated_binary");

// Use the operator
auto offspring = crossover_op->Crossover(parent1, parent2, rng);
```

## Mutation Operators

### Available Operators
- **BitFlipMutation**: Bit-flip mutation for binary representations
- **RandomResettingMutation**: Random resetting mutation
- **CreepMutation**: Small-step creep mutation
- **UniformMutation**: Uniform random mutation
- **GaussianMutation**: Gaussian (normal) mutation
- **SwapMutation**: Swap two elements
- **InversionMutation**: Invert a segment
- **InsertMutation**: Insert an element at new position
- **ScrambleMutation**: Scramble a segment
- **SelfAdaptiveMutation**: Self-adaptive mutation with strategy parameters
- **ListMutation**: Mutation for list-based representations

### Usage
```cpp
#include "mutation/all_mutation_operators.h"

// Create a mutation operator
auto mutation_op = mutation::CreateMutationOperator("gaussian");

// Use the operator
Individual mutated = mutation_op->Mutate(individual, rng);
```

## Selection Operators

### Available Operators
- **TournamentSelection**: Tournament selection
- **RouletteWheelSelection**: Fitness-proportionate selection
- **RankSelection**: Rank-based selection
- **StochasticUniversalSampling**: SUS selection
- **ElitismSelection**: Elitist selection

### Usage
```cpp
#include "selection-operator/all_selection_operators.h"

// Create a selection operator
auto selection_op = selection::CreateSelectionOperator("tournament");

// Use the operator
int selected_index = selection_op->Select(population, rng);
```

## Main Genetic Algorithm Framework

### GAParameters Structure
```cpp
struct GAParameters {
    int population_size = 100;
    int max_generations = 1000;
    double crossover_rate = 0.8;
    double mutation_rate = 0.1;
    double elitism_rate = 0.1;
    int tournament_size = 3;
    std::string crossover_type = "one_point";
    std::string mutation_type = "gaussian";
    std::string selection_type = "tournament";
    double convergence_threshold = 1e-6;
    int chromosome_length = 10;
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
};
```

### Creating a Custom Optimizer
```cpp
#include "module-GeneticAlgorithm.h"

class MyOptimizer : public GeneticAlgorithm::GeneticAlgorithmOptimizer {
public:
    explicit MyOptimizer(const GAParameters& params) 
        : GeneticAlgorithmOptimizer(params) {}
    
    double EvaluateFitness(const Individual& individual) override {
        // Implement your fitness function here
        double fitness = 0.0;
        // ... calculate fitness ...
        return fitness;
    }
};

// Use the optimizer
GAParameters params;
params.population_size = 100;
params.crossover_type = "simulated_binary";
params.mutation_type = "gaussian";

MyOptimizer optimizer(params);
Individual best = optimizer.Run();
```

## File Structure Details

### Crossover Directory
```
crossover/
├── base_crossover.h/cc               # Base class for all crossover operators
├── one_point_crossover.h/cc          # One-point crossover
├── two_point_crossover.h/cc          # Two-point crossover
├── multi_point_crossover.h/cc        # Multi-point crossover
├── uniform_crossover.h/cc            # Uniform crossover
├── uniform_k_vector_crossover.h/cc   # Uniform K-vector crossover
├── blend_crossover.h/cc              # Blend crossover (BLX-α)
├── simulated_binary_crossover.h/cc   # Simulated binary crossover
├── line_recombination.h/cc           # Line recombination
├── intermediate_recombination.h/cc   # Intermediate recombination
├── cut_and_crossfill_crossover.h/cc  # Cut-and-crossfill
├── partially_mapped_crossover.h/cc   # Partially mapped crossover
├── edge_crossover.h/cc               # Edge crossover
├── order_crossover.h/cc              # Order crossover
├── cycle_crossover.h/cc              # Cycle crossover
├── subtree_crossover.h/cc            # Subtree crossover
├── diploid_recombination.h/cc        # Diploid recombination
├── differential_evolution_crossover.h/cc # DE crossover
├── all_crossover_operators.h/cc      # Factory and utilities
└── README.md                         # Documentation
```

### Mutation Directory
```
mutation/
├── base_mutation.h/cc                # Base class for all mutation operators
├── bit_flip_mutation.h/cc            # Bit-flip mutation
├── random_resetting_mutation.h/cc    # Random resetting mutation
├── creep_mutation.h/cc               # Creep mutation
├── uniform_mutation.h/cc             # Uniform mutation
├── gaussian_mutation.h/cc            # Gaussian mutation
├── swap_mutation.h/cc                # Swap mutation
├── inversion_mutation.h/cc           # Inversion mutation
├── insert_mutation.h/cc              # Insert mutation
├── scramble_mutation.h/cc            # Scramble mutation
├── self_adaptive_mutation.h/cc       # Self-adaptive mutation
├── list_mutation.h/cc                # List mutation
├── all_mutation_operators.h/cc       # Factory and utilities
└── mutation.h/cc                     # Legacy compatibility
```

### Selection Directory
```
selection-operator/
├── base_selection.h/cc               # Base class for all selection operators
├── tournament_selection.h/cc         # Tournament selection
├── roulette_wheel_selection.h/cc     # Roulette wheel selection
├── rank_selection.h/cc               # Rank selection
├── stochastic_universal_sampling.h/cc # SUS selection
├── elitism_selection.h/cc            # Elitism selection
├── all_selection_operators.h/cc      # Factory and utilities
└── selection-operator.h/cc           # Legacy compatibility
```

## Key Features

### 1. Modular Design
- Each operator is in its own file with clear interfaces
- Easy to add new operators without modifying existing code
- Clean separation of concerns

### 2. Factory Pattern
- Operators can be created by string name
- Runtime operator selection
- Easy configuration from input files

### 3. Template-Based Design
- Type-safe implementations
- Efficient memory usage
- Compile-time optimizations

### 4. Extensibility
- Base classes provide clear interfaces for new operators
- Virtual functions allow polymorphic behavior
- Factory registration system for easy extension

### 5. Backward Compatibility
- Legacy function wrappers maintain existing API
- Gradual migration path for existing code
- No breaking changes to existing functionality

## Building and Testing

### Build Requirements
- C++11 or later
- MBDyn development environment
- Standard math libraries

### Compilation
The module is automatically built as part of the MBDyn build process. All source files are included in `Makefile.inc`.

### Testing
Run the comprehensive test:
```bash
cd /path/to/MBDyn/modules/module-GeneticAlgorithm
g++ -std=c++11 -I. test_complete_ga.cc *.cc crossover/*.cc mutation/*.cc selection-operator/*.cc -o test_ga
./test_ga
```

## Examples

### Example 1: Sphere Function Optimization
```cpp
class SphereOptimizer : public GeneticAlgorithmOptimizer {
public:
    explicit SphereOptimizer(const GAParameters& params) 
        : GeneticAlgorithmOptimizer(params) {}
    
    double EvaluateFitness(const Individual& individual) override {
        double sum = 0.0;
        for (double gene : individual.chromosome) {
            sum += gene * gene;
        }
        return -sum; // Negative because GA maximizes
    }
};
```

### Example 2: Multi-Objective Optimization
```cpp
// Set up parameters for multi-objective problem
GAParameters params;
params.population_size = 200;
params.crossover_type = "simulated_binary";
params.mutation_type = "gaussian";
params.selection_type = "tournament";

MyMultiObjectiveOptimizer optimizer(params);
Individual best = optimizer.Run();
```

## Performance Considerations

### Memory Usage
- Efficient vector-based representations
- Minimal memory allocation during evolution
- Smart pointer usage for operator management

### Computational Efficiency
- Inline functions for critical paths
- Template specializations where beneficial
- Optimized random number generation

### Scalability
- Population-based parallelization ready
- Operator-level parallelization support
- Memory-efficient for large problems

## Future Extensions

### Planned Features
1. **Multi-objective optimization operators**
2. **Parallel evaluation support**
3. **Adaptive parameter control**
4. **Advanced constraint handling**
5. **Hybrid algorithms (GA + local search)**

### Adding New Operators
1. Create operator class inheriting from base class
2. Implement required virtual methods
3. Add to factory registration
4. Update documentation
5. Add tests

## License
This module is part of MBDyn and follows the same licensing terms.

## Authors
- Genetic Algorithm Module Development Team
- MBDyn Development Team
- Dipartimento di Ingegneria Aerospaziale, Politecnico di Milano

## References
1. Goldberg, D.E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning
2. Deb, K. (2001). Multi-Objective Optimization using Evolutionary Algorithms
3. MBDyn User Manual and Developer Documentation
