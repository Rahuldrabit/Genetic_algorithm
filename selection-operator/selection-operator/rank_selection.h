#ifndef RANK_SELECTION_H
#define RANK_SELECTION_H

#include "base_selection.h"

/**
 * @brief Rank Selection Operator
 * 
 * In rank selection, individuals are sorted by fitness and selection probability
 * is based on rank rather than absolute fitness values. This helps avoid problems
 * with premature convergence when fitness differences are very large.
 * 
 * The selection pressure parameter controls how strongly better individuals are favored:
 * - pressure = 1.0: uniform selection (no pressure)
 * - pressure = 2.0: moderate pressure (default)
 * - pressure > 2.0: high pressure (strong bias toward best individuals)
 */
class RankSelection : public SelectionOperator {
private:
    double selection_pressure;
    
public:
    /**
     * @brief Construct a new Rank Selection object
     * @param pressure Selection pressure (default: 2.0)
     */
    explicit RankSelection(double pressure = 2.0);
    
    /**
     * @brief Select individuals using rank-based selection
     * @param population The population to select from
     * @param count Number of individuals to select
     * @return Vector of selected individuals
     */
    std::vector<Individual> select(const std::vector<Individual>& population, size_t count) override;
    
    /**
     * @brief Get the selection pressure
     * @return Current selection pressure value
     */
    double getSelectionPressure() const { return selection_pressure; }
    
    /**
     * @brief Set the selection pressure
     * @param pressure New selection pressure value (must be >= 1.0)
     */
    void setSelectionPressure(double pressure);
    
    /**
     * @brief Legacy function for backward compatibility
     * @param Population Population to select from
     * @param NumSelections Number of selections to make
     * @return Vector of selected indices
     */
    static std::vector<unsigned int> selectIndices(std::vector<Individual>& Population, 
                                                  unsigned int NumSelections);
};

// Legacy function declaration for backward compatibility
std::vector<unsigned int> RankSelectionLegacy(std::vector<Individual>& Population, unsigned int NumSelections);

#endif // RANK_SELECTION_H
