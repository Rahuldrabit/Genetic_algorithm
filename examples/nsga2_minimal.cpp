#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "ga/algorithms/moea/nsga2.hpp"

class PointGenome : public ga::IGenome {
public:
    PointGenome(double x, double y) : x_(x), y_(y) {}

    std::unique_ptr<ga::IGenome> clone() const override {
        return std::make_unique<PointGenome>(*this);
    }

    std::string encodingName() const override {
        return "point2d";
    }

    double x() const { return x_; }
    double y() const { return y_; }
    void set(double x, double y) {
        x_ = x;
        y_ = y;
    }

private:
    double x_;
    double y_;
};

static PointGenome* asPoint(ga::Individual& individual) {
    auto* genome = dynamic_cast<PointGenome*>(individual.genome.get());
    if (!genome) {
        throw std::runtime_error("Expected PointGenome in individual");
    }
    return genome;
}

static const PointGenome* asPoint(const ga::Individual& individual) {
    auto* genome = dynamic_cast<const PointGenome*>(individual.genome.get());
    if (!genome) {
        throw std::runtime_error("Expected PointGenome in individual");
    }
    return genome;
}

int main() {
    constexpr std::size_t kPopulation = 80;
    constexpr std::size_t kGenerations = 60;
    constexpr double kInitMin = -4.0;
    constexpr double kInitMax = 4.0;
    constexpr double kMutSigma = 0.15;

    std::mt19937 init_rng(42);
    std::uniform_real_distribution<double> init_dist(kInitMin, kInitMax);

    std::vector<ga::Individual> initial;
    initial.reserve(kPopulation);
    for (std::size_t i = 0; i < kPopulation; ++i) {
        ga::Individual ind;
        ind.genome = std::make_unique<PointGenome>(init_dist(init_rng), init_dist(init_rng));
        initial.push_back(std::move(ind));
    }

    ga::moea::Nsga2Config cfg;
    cfg.populationSize = kPopulation;
    cfg.generations = kGenerations;
    cfg.seed = 7;

    ga::moea::Nsga2 nsga2(cfg);

    auto evaluate = [](std::vector<ga::Individual>& population) {
        for (auto& ind : population) {
            PointGenome* g = asPoint(ind);
            const double x = g->x();
            const double y = g->y();

            // Two-objective minimization problem with conflicting optima.
            const double f1 = x * x + y * y;
            const double dx = x - 1.5;
            const double dy = y + 1.5;
            const double f2 = dx * dx + dy * dy;

            ind.evaluation.objectives = {f1, f2};
            ind.evaluation.feasible = true;
            ind.evaluation.penalty = 0.0;
        }
    };

    auto reproduce = [=](const std::vector<ga::Individual>& parents,
                         const std::vector<std::size_t>& mating,
                         std::mt19937& rng) {
        std::normal_distribution<double> noise(0.0, kMutSigma);
        std::vector<ga::Individual> offspring;
        offspring.reserve(mating.size());

        for (std::size_t idx : mating) {
            const PointGenome* parent = asPoint(parents[idx]);

            double x = parent->x() + noise(rng);
            double y = parent->y() + noise(rng);
            x = std::clamp(x, kInitMin, kInitMax);
            y = std::clamp(y, kInitMin, kInitMax);

            ga::Individual child;
            child.genome = std::make_unique<PointGenome>(x, y);
            offspring.push_back(std::move(child));
        }

        return offspring;
    };

    auto result = nsga2.run(std::move(initial), evaluate, reproduce);
    auto fronts = nsga2.nonDominatedSort(result.population);

    std::cout << "NSGA-II completed\n";
    std::cout << "History points: " << result.history.size() << "\n";
    std::cout << "Final first-front size: " << (fronts.empty() ? 0 : fronts.front().size()) << "\n";
    std::cout << "Sample Pareto points (x, y -> f1, f2):\n";

    std::size_t to_print = 0;
    if (!fronts.empty()) {
        for (std::size_t idx : fronts.front()) {
            const PointGenome* g = asPoint(result.population[idx]);
            const auto& obj = result.population[idx].evaluation.objectives;
            std::cout << "  (" << g->x() << ", " << g->y() << ") -> ("
                      << obj[0] << ", " << obj[1] << ")\n";
            ++to_print;
            if (to_print >= 8) {
                break;
            }
        }
    }

    return 0;
}
