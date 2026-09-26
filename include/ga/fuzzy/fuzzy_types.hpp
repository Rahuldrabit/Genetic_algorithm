#pragma once

#include <cstddef>
#include <string>

namespace ga {
namespace fuzzy {

enum class TNormType {
    Minimum,          // Zadeh min(a, b)
    AlgebraicProduct, // a * b
    Lukasiewicz,      // max(0, a + b - 1)
    DrasticProduct,   // b if a==1, a if b==1, else 0
    HamacherProduct,  // (a*b) / (gamma + (1-gamma)*(a+b-a*b)), default gamma=0.5
    EinsteinProduct   // (a*b) / (2 - (a + b - a*b))
};

enum class SNormType {
    Maximum,          // Zadeh max(a, b)
    AlgebraicSum,     // a + b - a*b
    Lukasiewicz,      // min(1, a + b)
    DrasticSum,       // b if a==0, a if b==0, else 1
    HamacherSum,      // (a + b - (2-gamma)*a*b) / (1 - (1-gamma)*a*b)
    EinsteinSum       // (a + b) / (1 + a*b)
};

enum class ComplementType {
    Standard, // 1 - a
    Sugeno,   // (1 - a) / (1 + lambda * a), lambda > -1
    Yager     // (1 - a^w)^(1/w), w > 0
};

enum class DefuzzMethod {
    Centroid,        // Center of Gravity / Area (COG)
    Bisector,        // Divides area into equal halves
    MeanOfMaxima,    // MOM
    SmallestOfMaxima,// SOM
    LargestOfMaxima, // LOM
    WeightedAverage, // For Sugeno FIS
    MOM = MeanOfMaxima,
    SOM = SmallestOfMaxima,
    LOM = LargestOfMaxima
};

struct Interval {
    double min = 0.0;
    double max = 1.0;

    bool contains(double x) const noexcept {
        return x >= min && x <= max;
    }

    double span() const noexcept {
        return max - min;
    }

    double center() const noexcept {
        return 0.5 * (min + max);
    }
};

} // namespace fuzzy
} // namespace ga
