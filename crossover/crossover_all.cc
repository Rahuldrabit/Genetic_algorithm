#include "crossover_all.h"

std::unique_ptr<CrossoverOperator> createCrossoverOperator(const std::string& type, unsigned seed) {
    if (type == "one_point") {
        return std::make_unique<OnePointCrossover>(seed);
    } else if (type == "two_point") {
        return std::make_unique<TwoPointCrossover>(seed);
    } else if (type == "uniform") {
        return std::make_unique<UniformCrossover>(0.5, seed);
    } else if (type == "arithmetic" || type == "intermediate") {
        return std::make_unique<IntermediateRecombination>(0.5, seed);
    } else if (type == "blend") {
        return std::make_unique<BlendCrossover>(0.5, seed);
    } else if (type == "sbx") {
        return std::make_unique<SimulatedBinaryCrossover>(2.0, seed);
    } else if (type == "order_crossover" || type == "ox") {
        return std::make_unique<OrderCrossover>(seed);
    } else if (type == "pmx") {
        return std::make_unique<PartiallyMappedCrossover>(seed);
    } else if (type == "cycle") {
        return std::make_unique<CycleCrossover>(seed);
    } else {
        // Default to one-point crossover
        return std::make_unique<OnePointCrossover>(seed);
    }
}
