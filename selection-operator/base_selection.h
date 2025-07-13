#ifndef BASE_SELECTION_H
#define BASE_SELECTION_H

#include <vector>
#include <string>
#include <random>

// Forward declarations
struct Individual {
    double fitness;
    std::vector<double> genes;
    int age = 0;
    
    Individual() : fitness(0.0) {}
    Individual(double f) : fitness(f) {}
    Individual(const std::vector<double>& g, double f) : fitness(f), genes(g) {}
    
    bool operator<(const Individual& other) const {
        return fitness < other.fitness;
    }
};

struct MultiObjectiveIndividual {
    std::vector<double> objectives;
    std::vector<double> genes;
    double crowdingDistance = 0.0;
    int rank = 0;
    int dominationCount = 0;
    std::vector<int> dominatedSolutions;
    int age = 0;
};

// Selection operator base class
class SelectionOperator {
protected:
    std::string name;
    mutable std::mt19937 rng;
    mutable size_t operation_count;
    
public:
    SelectionOperator(const std::string& op_name, unsigned seed = std::random_device{}()) 
        : name(op_name), rng(seed), operation_count(0) {}
    virtual ~SelectionOperator() = default;
    
    virtual std::vector<Individual> select(const std::vector<Individual>& population, 
                                         size_t count) = 0;
    
    // Statistics and utility methods
    std::string getName() const { return name; }
    size_t getOperationCount() const { return operation_count; }
    void resetStatistics() { operation_count = 0; }
    void setSeed(unsigned seed) { rng.seed(seed); }
};

#endif // BASE_SELECTION_H
