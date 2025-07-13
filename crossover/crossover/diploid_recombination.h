#ifndef DIPLOID_RECOMBINATION_H
#define DIPLOID_RECOMBINATION_H

#include "base_crossover.h"

class DiploidRecombination : public CrossoverOperator {
public:
    DiploidRecombination(unsigned seed = std::random_device{}()) 
        : CrossoverOperator("DiploidRecombination", seed) {}
    
    using DiploidChromosome = std::pair<BitString, BitString>;
    DiploidChromosome crossover(const DiploidChromosome& parent1, const DiploidChromosome& parent2);
    
private:
    BitString formGamete(const DiploidChromosome& parent);
};

#endif // DIPLOID_RECOMBINATION_H
