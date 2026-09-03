#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ga/metaheuristics.hpp"

namespace {

double rastrigin(const std::vector<double>& x) {
    const double pi = std::acos(-1.0);
    double value = 10.0 * static_cast<double>(x.size());
    for (double xi : x) {
        value += xi * xi - 10.0 * std::cos(2.0 * pi * xi);
    }
    return 1.0 / (1.0 + value);
}

template <typename Function>
void measure(const std::string& name, Function&& run) {
    const auto begin = std::chrono::steady_clock::now();
    const auto result = run();
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin);
    std::cout << std::left << std::setw(24) << name << std::right << std::setw(12)
              << std::fixed << std::setprecision(3) << elapsed.count() << std::setw(16)
              << result.evaluations << std::setw(18) << std::setprecision(8)
              << result.bestFitness << '\n';
}

} // namespace

int main() {
    auto controller = std::make_shared<ga::fuzzy::FuzzyAdaptiveController>();
    ga::metaheuristics::SearchConfig search;
    search.populationSize = 80;
    search.iterations = 150;
    search.dimension = 30;
    search.bounds = {-5.12, 5.12};
    search.seed = 42;

    std::cout << std::left << std::setw(24) << "algorithm" << std::right
              << std::setw(12) << "ms" << std::setw(16) << "evaluations"
              << std::setw(18) << "best fitness" << '\n';

    for (const auto variant : {ga::pso::PsoVariant::GlobalBest,
                               ga::pso::PsoVariant::LocalBest,
                               ga::pso::PsoVariant::Constriction,
                               ga::pso::PsoVariant::BareBones,
                               ga::pso::PsoVariant::FullyInformed,
                               ga::pso::PsoVariant::QuantumBehaved}) {
        ga::pso::PsoConfig config;
        config.search = search;
        config.variant = variant;
        config.controller = controller;
        ga::pso::ParticleSwarmOptimizer optimizer(config);
        const std::string name = optimizer.name();
        measure(name, [&] { return optimizer.optimize(rastrigin); });
    }

    ga::aco::AcorConfig acorConfig;
    acorConfig.search = search;
    acorConfig.archiveSize = 80;
    acorConfig.sampleCount = 40;
    acorConfig.controller = controller;
    ga::aco::ContinuousAntColonyOptimizer acor(acorConfig);
    measure(acor.name(), [&] { return acor.optimize(rastrigin); });

    ga::gsa::GsaConfig gsaConfig;
    gsaConfig.search = search;
    gsaConfig.controller = controller;
    ga::gsa::GravitationalSearchOptimizer gsa(gsaConfig);
    measure(gsa.name(), [&] { return gsa.optimize(rastrigin); });
    return 0;
}
