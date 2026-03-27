#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ga/algorithms/moea/nsga2.hpp"

namespace ga {
namespace moea {

class Nsga3 {
public:
    explicit Nsga3(Nsga2Config cfg = {}) : nsga2_(cfg) {}

    static std::vector<std::vector<double>>
    generateDasDennisReferencePoints(std::size_t objectiveCount,
                                     std::size_t divisions) {
        if (objectiveCount == 0) {
            throw std::invalid_argument("NSGA-III requires at least one objective");
        }
        if (divisions == 0) {
            throw std::invalid_argument("NSGA-III reference point divisions must be > 0");
        }

        std::vector<std::vector<double>> points;
        std::vector<double> current;
        current.reserve(objectiveCount);
        generateReferencePointsRecursive(objectiveCount, divisions, divisions, current, points);
        return points;
    }

    std::vector<std::vector<std::size_t>>
    nonDominatedSort(const std::vector<ga::Individual>& population) const {
        return nsga2_.nonDominatedSort(population);
    }

    std::vector<ga::Individual>
    environmentalSelect(const std::vector<ga::Individual>& combined,
                        std::size_t targetSize,
                        const std::vector<std::vector<double>>& referencePoints) const {
        if (targetSize == 0 || combined.empty()) {
            return {};
        }
        if (targetSize > combined.size()) {
            throw std::invalid_argument("NSGA-III target size exceeds combined population size");
        }
        if (referencePoints.empty()) {
            return nsga2_.environmentalSelect(combined, targetSize);
        }

        validateObjectives(combined);
        validateReferencePoints(combined, referencePoints);

        const auto fronts = nsga2_.nonDominatedSort(combined);
        std::vector<std::size_t> selected;
        selected.reserve(targetSize);

        const auto normalization = computeNormalization(combined, fronts);

        std::vector<std::size_t> partialFront;
        for (const auto& front : fronts) {
            if (selected.size() + front.size() <= targetSize) {
                selected.insert(selected.end(), front.begin(), front.end());
            } else {
                partialFront = front;
                break;
            }
        }

        if (!partialFront.empty() && selected.size() < targetSize) {
            auto nicheCounts = computeNicheCounts(selected, combined, referencePoints, normalization);
            auto nicheBuckets = buildNicheBuckets(partialFront, combined, referencePoints, normalization);

            while (selected.size() < targetSize) {
                const std::size_t niche = pickLeastCrowdedNiche(nicheCounts, nicheBuckets);
                if (niche == referencePoints.size()) {
                    break;
                }
                const std::size_t winner = pickBestCandidateInNiche(nicheBuckets[niche]);
                selected.push_back(nicheBuckets[niche][winner].index);
                nicheBuckets[niche].erase(nicheBuckets[niche].begin() +
                                          static_cast<std::ptrdiff_t>(winner));
                nicheCounts[niche] += 1;
            }

            if (selected.size() < targetSize) {
                std::vector<Candidate> remaining;
                for (const auto& bucket : nicheBuckets) {
                    remaining.insert(remaining.end(), bucket.begin(), bucket.end());
                }
                std::sort(remaining.begin(), remaining.end(), [](const Candidate& a, const Candidate& b) {
                    if (a.distance != b.distance) {
                        return a.distance < b.distance;
                    }
                    return a.index < b.index;
                });
                for (const auto& c : remaining) {
                    if (selected.size() >= targetSize) {
                        break;
                    }
                    selected.push_back(c.index);
                }
            }
        }

        if (selected.size() < targetSize) {
            const auto fallback = nsga2_.environmentalSelect(combined, targetSize);
            return fallback;
        }

        std::vector<ga::Individual> out;
        out.reserve(targetSize);
        for (std::size_t idx : selected) {
            out.push_back(combined[idx]);
        }
        return out;
    }

private:
    struct Normalization {
        std::vector<double> ideal;
        std::vector<double> intercepts;
    };

    struct Candidate {
        std::size_t index = 0;
        std::size_t niche = 0;
        double distance = std::numeric_limits<double>::infinity();
    };

    static void generateReferencePointsRecursive(std::size_t dimensionsRemaining,
                                                 std::size_t divisions,
                                                 std::size_t unitsRemaining,
                                                 std::vector<double>& current,
                                                 std::vector<std::vector<double>>& out) {
        if (dimensionsRemaining == 1) {
            current.push_back(static_cast<double>(unitsRemaining) / static_cast<double>(divisions));
            out.push_back(current);
            current.pop_back();
            return;
        }

        for (std::size_t u = 0; u <= unitsRemaining; ++u) {
            current.push_back(static_cast<double>(u) / static_cast<double>(divisions));
            generateReferencePointsRecursive(dimensionsRemaining - 1,
                                             divisions,
                                             unitsRemaining - u,
                                             current,
                                             out);
            current.pop_back();
        }
    }

    static void validateObjectives(const std::vector<ga::Individual>& population) {
        if (population.empty()) {
            throw std::invalid_argument("NSGA-III requires a non-empty population");
        }
        const std::size_t dims = population.front().evaluation.objectives.size();
        if (dims == 0) {
            throw std::invalid_argument("NSGA-III requires objective values");
        }
        for (const auto& ind : population) {
            if (ind.evaluation.objectives.size() != dims) {
                throw std::invalid_argument(
                    "NSGA-III objective dimensionality mismatch across individuals");
            }
        }
    }

    static void validateReferencePoints(const std::vector<ga::Individual>& population,
                                        const std::vector<std::vector<double>>& referencePoints) {
        const std::size_t dims = population.front().evaluation.objectives.size();
        for (const auto& rp : referencePoints) {
            if (rp.size() != dims) {
                throw std::invalid_argument("NSGA-III reference point dimensionality mismatch");
            }
            double norm = 0.0;
            for (double v : rp) {
                norm += v * v;
            }
            if (norm <= 0.0) {
                throw std::invalid_argument(
                    "NSGA-III reference points must have non-zero norm");
            }
        }
    }

    static std::vector<double> translatedObjectives(const ga::Individual& ind,
                                                    const std::vector<double>& ideal) {
        std::vector<double> translated;
        translated.reserve(ind.evaluation.objectives.size());
        for (std::size_t j = 0; j < ind.evaluation.objectives.size(); ++j) {
            translated.push_back(std::max(0.0, ind.evaluation.objectives[j] - ideal[j]));
        }
        return translated;
    }

    static std::vector<double> idealPoint(const std::vector<ga::Individual>& population) {
        const std::size_t dims = population.front().evaluation.objectives.size();
        std::vector<double> ideal(dims, std::numeric_limits<double>::infinity());

        for (const auto& ind : population) {
            for (std::size_t j = 0; j < dims; ++j) {
                ideal[j] = std::min(ideal[j], ind.evaluation.objectives[j]);
            }
        }
        return ideal;
    }

    static std::vector<double> nadirPoint(const std::vector<ga::Individual>& population,
                                          const std::vector<double>& ideal) {
        const std::size_t dims = ideal.size();
        std::vector<double> nadir(dims, 0.0);
        for (const auto& ind : population) {
            for (std::size_t j = 0; j < dims; ++j) {
                nadir[j] = std::max(nadir[j], ind.evaluation.objectives[j] - ideal[j]);
            }
        }
        for (double& v : nadir) {
            if (v <= 1e-12) {
                v = 1.0;
            }
        }
        return nadir;
    }

    static double asfValue(const std::vector<double>& point,
                           const std::vector<double>& weights) {
        double value = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < point.size(); ++j) {
            value = std::max(value, point[j] / weights[j]);
        }
        return value;
    }

    static std::vector<std::vector<double>>
    findExtremePoints(const std::vector<ga::Individual>& population,
                      const std::vector<std::vector<std::size_t>>& fronts,
                      const std::vector<double>& ideal) {
        const std::size_t dims = ideal.size();
        const std::vector<std::size_t>* firstFront = nullptr;
        if (!fronts.empty() && !fronts.front().empty()) {
            firstFront = &fronts.front();
        }

        std::vector<std::vector<double>> extreme(dims, std::vector<double>(dims, 0.0));
        for (std::size_t m = 0; m < dims; ++m) {
            std::vector<double> w(dims, 1e-6);
            w[m] = 1.0;

            double best = std::numeric_limits<double>::infinity();
            std::size_t bestIdx = 0;
            if (firstFront) {
                for (std::size_t idx : *firstFront) {
                    const auto t = translatedObjectives(population[idx], ideal);
                    const double a = asfValue(t, w);
                    if (a < best) {
                        best = a;
                        bestIdx = idx;
                    }
                }
            } else {
                for (std::size_t idx = 0; idx < population.size(); ++idx) {
                    const auto t = translatedObjectives(population[idx], ideal);
                    const double a = asfValue(t, w);
                    if (a < best) {
                        best = a;
                        bestIdx = idx;
                    }
                }
            }
            extreme[m] = translatedObjectives(population[bestIdx], ideal);
        }

        return extreme;
    }

    static std::vector<double> solveLinearSystem(std::vector<std::vector<double>> a,
                                                 std::vector<double> b) {
        const std::size_t n = a.size();
        for (std::size_t col = 0; col < n; ++col) {
            std::size_t pivot = col;
            double maxAbs = std::fabs(a[col][col]);
            for (std::size_t r = col + 1; r < n; ++r) {
                const double v = std::fabs(a[r][col]);
                if (v > maxAbs) {
                    maxAbs = v;
                    pivot = r;
                }
            }
            if (maxAbs <= 1e-12) {
                return {};
            }
            if (pivot != col) {
                std::swap(a[pivot], a[col]);
                std::swap(b[pivot], b[col]);
            }

            const double diag = a[col][col];
            for (std::size_t j = col; j < n; ++j) {
                a[col][j] /= diag;
            }
            b[col] /= diag;

            for (std::size_t r = 0; r < n; ++r) {
                if (r == col) {
                    continue;
                }
                const double factor = a[r][col];
                if (std::fabs(factor) <= 1e-16) {
                    continue;
                }
                for (std::size_t j = col; j < n; ++j) {
                    a[r][j] -= factor * a[col][j];
                }
                b[r] -= factor * b[col];
            }
        }
        return b;
    }

    static std::vector<double> computeIntercepts(const std::vector<std::vector<double>>& extreme,
                                                 const std::vector<double>& nadir) {
        const std::size_t dims = extreme.size();
        std::vector<double> ones(dims, 1.0);
        auto lambda = solveLinearSystem(extreme, ones);
        if (lambda.size() != dims) {
            return nadir;
        }

        std::vector<double> intercepts(dims, 0.0);
        for (std::size_t j = 0; j < dims; ++j) {
            if (std::fabs(lambda[j]) <= 1e-12) {
                return nadir;
            }
            const double value = 1.0 / lambda[j];
            if (!std::isfinite(value) || value <= 1e-12) {
                return nadir;
            }
            intercepts[j] = value;
        }

        return intercepts;
    }

    static Normalization computeNormalization(
        const std::vector<ga::Individual>& population,
        const std::vector<std::vector<std::size_t>>& fronts) {
        Normalization norm;
        norm.ideal = idealPoint(population);
        const auto nadir = nadirPoint(population, norm.ideal);
        const auto extreme = findExtremePoints(population, fronts, norm.ideal);
        norm.intercepts = computeIntercepts(extreme, nadir);
        return norm;
    }

    static std::vector<double> normalizeObjectives(const ga::Individual& ind,
                                                   const Normalization& norm) {
        auto translated = translatedObjectives(ind, norm.ideal);
        std::vector<double> normalized;
        normalized.reserve(translated.size());
        for (std::size_t j = 0; j < translated.size(); ++j) {
            const double denom = std::max(1e-12, norm.intercepts[j]);
            normalized.push_back(translated[j] / denom);
        }
        return normalized;
    }

    static Candidate associateToReferencePoint(const ga::Individual& ind,
                                               std::size_t index,
                                               const std::vector<std::vector<double>>& referencePoints,
                                               const Normalization& norm) {
        const auto p = normalizeObjectives(ind, norm);

        Candidate c;
        c.index = index;
        for (std::size_t r = 0; r < referencePoints.size(); ++r) {
            const double d = perpendicularDistance(p, referencePoints[r]);
            if (d < c.distance) {
                c.distance = d;
                c.niche = r;
            }
        }
        return c;
    }

    static std::vector<std::size_t>
    computeNicheCounts(const std::vector<std::size_t>& selected,
                       const std::vector<ga::Individual>& population,
                       const std::vector<std::vector<double>>& referencePoints,
                       const Normalization& norm) {
        std::vector<std::size_t> counts(referencePoints.size(), 0);
        for (std::size_t idx : selected) {
            const auto assoc = associateToReferencePoint(population[idx], idx, referencePoints, norm);
            counts[assoc.niche] += 1;
        }
        return counts;
    }

    static std::vector<std::vector<Candidate>>
    buildNicheBuckets(const std::vector<std::size_t>& front,
                      const std::vector<ga::Individual>& population,
                      const std::vector<std::vector<double>>& referencePoints,
                      const Normalization& norm) {
        std::vector<std::vector<Candidate>> buckets(referencePoints.size());
        for (std::size_t idx : front) {
            auto assoc = associateToReferencePoint(population[idx], idx, referencePoints, norm);
            buckets[assoc.niche].push_back(std::move(assoc));
        }
        return buckets;
    }

    static std::size_t pickLeastCrowdedNiche(const std::vector<std::size_t>& nicheCounts,
                                             const std::vector<std::vector<Candidate>>& buckets) {
        std::size_t chosen = buckets.size();
        std::size_t minCount = std::numeric_limits<std::size_t>::max();
        for (std::size_t n = 0; n < buckets.size(); ++n) {
            if (buckets[n].empty()) {
                continue;
            }
            if (nicheCounts[n] < minCount) {
                minCount = nicheCounts[n];
                chosen = n;
            }
        }
        return chosen;
    }

    static std::size_t pickBestCandidateInNiche(const std::vector<Candidate>& bucket) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < bucket.size(); ++i) {
            if (bucket[i].distance < bucket[best].distance ||
                (bucket[i].distance == bucket[best].distance &&
                 bucket[i].index < bucket[best].index)) {
                best = i;
            }
        }
        return best;
    }

    static double perpendicularDistance(const std::vector<double>& point,
                                        const std::vector<double>& reference) {
        double dot = 0.0;
        double rr = 0.0;
        for (std::size_t i = 0; i < point.size(); ++i) {
            dot += point[i] * reference[i];
            rr += reference[i] * reference[i];
        }
        if (rr <= 1e-12) {
            return std::numeric_limits<double>::infinity();
        }
        const double scale = std::max(0.0, dot / rr);

        double dist2 = 0.0;
        for (std::size_t i = 0; i < point.size(); ++i) {
            const double proj = scale * reference[i];
            const double delta = point[i] - proj;
            dist2 += delta * delta;
        }
        return std::sqrt(dist2);
    }

    Nsga2 nsga2_;
};

} // namespace moea
} // namespace ga
