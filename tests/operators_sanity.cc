#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <utility>

// Crossover operators
#include "ga/crossover/base_crossover.h"
#include "ga/crossover/one_point_crossover.h"
#include "ga/crossover/two_point_crossover.h"
#include "ga/crossover/multi_point_crossover.h"
#include "ga/crossover/uniform_crossover.h"
#include "ga/crossover/uniform_k_vector_crossover.h"
#include "ga/crossover/blend_crossover.h"
#include "ga/crossover/simulated_binary_crossover.h"
#include "ga/crossover/line_recombination.h"
#include "ga/crossover/intermediate_recombination.h"
#include "ga/crossover/cut_and_crossfill_crossover.h"
#include "ga/crossover/partially_mapped_crossover.h"
#include "ga/crossover/edge_crossover.h"
#include "ga/crossover/order_crossover.h"
#include "ga/crossover/cycle_crossover.h"
#include "ga/crossover/subtree_crossover.h"
#include "ga/crossover/diploid_recombination.h"
#include "ga/crossover/differential_evolution_crossover.h"

// Mutation operators
#include "ga/mutation/base_mutation.h"
#include "ga/mutation/bit_flip_mutation.h"
#include "ga/mutation/random_resetting_mutation.h"
#include "ga/mutation/creep_mutation.h"
#include "ga/mutation/uniform_mutation.h"
#include "ga/mutation/gaussian_mutation.h"
#include "ga/mutation/swap_mutation.h"
#include "ga/mutation/inversion_mutation.h"
#include "ga/mutation/insert_mutation.h"
#include "ga/mutation/scramble_mutation.h"
#include "ga/mutation/list_mutation.h"

// Selection operators
#include "ga/selection/base_selection.h"
#include "ga/selection/tournament_selection.h"
#include "ga/selection/roulette_wheel_selection.h"
#include "ga/selection/rank_selection.h"
#include "ga/selection/stochastic_universal_sampling.h"
#include "ga/selection/elitism_selection.h"

using std::cout;
using std::endl;

static void print_status(const std::string& name, bool ok, const std::string& note = "") {
    if (ok) cout << "[PASS] " << name << (note.empty()?"":" - "+note) << endl;
    else cout << "[FAIL] " << name << (note.empty()?"":" - "+note) << endl;
}

static std::vector<double> make_real(int n) {
    std::vector<double> v(n);
    for (int i=0;i<n;++i) v[i] = (i%5) - 2.0; // -2,-1,0,1,2
    return v;
}
static std::vector<bool> make_bits(int n) {
    std::vector<bool> v(n);
    for (int i=0;i<n;++i) v[i] = (i%2)==0;
    return v;
}
static std::vector<int> make_ints(int n) {
    std::vector<int> v(n);
    for (int i=0;i<n;++i) v[i] = i;
    return v;
}
static std::vector<int> make_perm(int n) { return make_ints(n); }

int main(){
    bool all_ok = true;

    // Crossovers (real)
    {
        auto a = make_real(10), b = make_real(10);
        try { OnePointCrossover op; auto c = op.crossover(a,b); print_status("OnePointCrossover<Real>", c.first.size()==10 && c.second.size()==10); } catch(const std::exception& e){ print_status("OnePointCrossover<Real>", false, e.what()); all_ok=false; }
        try { TwoPointCrossover op; auto c = op.crossover(a,b); print_status("TwoPointCrossover<Real>", c.first.size()==10 && c.second.size()==10); } catch(const std::exception& e){ print_status("TwoPointCrossover<Real>", false, e.what()); all_ok=false; }
        try { UniformCrossover op; auto c = op.crossover(a,b); print_status("UniformCrossover<Real>", c.first.size()==10 && c.second.size()==10); } catch(const std::exception& e){ print_status("UniformCrossover<Real>", false, e.what()); all_ok=false; }
        try { MultiPointCrossover op(3); auto c = op.crossover(a,b); print_status("MultiPointCrossover<Real>", c.first.size()==10 && c.second.size()==10); } catch(const std::exception& e){ print_status("MultiPointCrossover<Real>", false, e.what()); all_ok=false; }
        try { BlendCrossover op(0.5); auto c = op.crossover(a,b); print_status("BlendCrossover<Real>", c.first.size()==10 && c.second.size()==10); } catch(const std::exception& e){ print_status("BlendCrossover<Real>", false, e.what()); all_ok=false; }
        try { SimulatedBinaryCrossover op(2.0); auto c = op.crossover(a,b); print_status("SBX<Real>", c.first.size()==10 && c.second.size()==10); } catch(const std::exception& e){ print_status("SBX<Real>", false, e.what()); all_ok=false; }
        try { LineRecombination op; auto c = op.crossover(a,b); print_status("LineRecombination<Real>", c.first.size()==10 && c.second.size()==10); } catch(const std::exception& e){ print_status("LineRecombination<Real>", false, e.what()); all_ok=false; }
        try { IntermediateRecombination op; auto c = op.crossover(a,b); print_status("IntermediateRecombination<Real>", c.first.size()==10 && c.second.size()==10); } catch(const std::exception& e){ print_status("IntermediateRecombination<Real>", false, e.what()); all_ok=false; }
        try { DifferentialEvolutionCrossover op; auto t = a, m=b; auto c = op.performCrossover(t,m); print_status("DifferentialEvolutionCrossover<Real>", c.size()==10); } catch(const std::exception& e){ print_status("DifferentialEvolutionCrossover<Real>", false, e.what()); all_ok=false; }
    }

    // Crossovers (binary)
    {
        auto a = make_bits(12), b = make_bits(12);
        try { OnePointCrossover op; auto c = op.crossover(a,b); print_status("OnePointCrossover<Bit>", c.first.size()==12 && c.second.size()==12); } catch(const std::exception& e){ print_status("OnePointCrossover<Bit>", false, e.what()); all_ok=false; }
        try { TwoPointCrossover op; auto c = op.crossover(a,b); print_status("TwoPointCrossover<Bit>", c.first.size()==12 && c.second.size()==12); } catch(const std::exception& e){ print_status("TwoPointCrossover<Bit>", false, e.what()); all_ok=false; }
        try { UniformCrossover op; auto c = op.crossover(a,b); print_status("UniformCrossover<Bit>", c.first.size()==12 && c.second.size()==12); } catch(const std::exception& e){ print_status("UniformCrossover<Bit>", false, e.what()); all_ok=false; }
    }

    // Crossovers (integer vectors)
    {
        auto a = make_ints(9), b = make_ints(9);
        try { OnePointCrossover op; auto c = op.crossover(a,b); print_status("OnePointCrossover<Int>", c.first.size()==9 && c.second.size()==9); } catch(const std::exception& e){ print_status("OnePointCrossover<Int>", false, e.what()); all_ok=false; }
        try { TwoPointCrossover op; auto c = op.crossover(a,b); print_status("TwoPointCrossover<Int>", c.first.size()==9 && c.second.size()==9); } catch(const std::exception& e){ print_status("TwoPointCrossover<Int>", false, e.what()); all_ok=false; }
        try { UniformCrossover op; auto c = op.crossover(a,b); print_status("UniformCrossover<Int>", c.first.size()==9 && c.second.size()==9); } catch(const std::exception& e){ print_status("UniformCrossover<Int>", false, e.what()); all_ok=false; }
    }

    // Permutation crossovers
    {
        auto a = make_perm(10), b = make_perm(10);
        auto is_perm = [](const std::vector<int>& v){
            std::vector<int> s = v; std::sort(s.begin(), s.end());
            for (int i=0;i<(int)s.size();++i) { if (s[i]!=i) return false; } return true;
        };
        try { OrderCrossover op; auto c = op.crossover(a,b); bool ok = is_perm(c.first) && is_perm(c.second); print_status("OrderCrossover<Perm>", ok); if(!ok) all_ok=false; } catch(const std::exception& e){ print_status("OrderCrossover<Perm>", false, e.what()); all_ok=false; }
        try { PartiallyMappedCrossover op; auto c = op.crossover(a,b); bool ok = is_perm(c.first) && is_perm(c.second); print_status("PMX<Perm>", ok); if(!ok) all_ok=false; } catch(const std::exception& e){ print_status("PMX<Perm>", false, e.what()); all_ok=false; }
        try { CycleCrossover op; auto c = op.crossover(a,b); bool ok = is_perm(c.first) && is_perm(c.second); print_status("CycleCrossover<Perm>", ok); if(!ok) all_ok=false; } catch(const std::exception& e){ print_status("CycleCrossover<Perm>", false, e.what()); all_ok=false; }
        try { EdgeCrossover op; auto c = op.crossover(a,b); bool ok = is_perm(c.first) && is_perm(c.second); print_status("EdgeCrossover<Perm>", ok); if(!ok) all_ok=false; } catch(const std::exception& e){ print_status("EdgeCrossover<Perm>", false, e.what()); all_ok=false; }
        try { CutAndCrossfillCrossover op; auto c = op.crossover(a,b); bool ok = is_perm(c.first) && is_perm(c.second); print_status("CutAndCrossfill<Perm>", ok); if(!ok) all_ok=false; } catch(const std::exception& e){ print_status("CutAndCrossfill<Perm>", false, e.what()); all_ok=false; }
    }

    // Mutation real-valued
    {
        auto x = make_real(10);
        std::vector<double> lo(10,-5), hi(10,5);
        try { GaussianMutation op; op.mutate(x, 0.5, 0.1, lo, hi); print_status("GaussianMutation<Real>", x.size()==10); } catch(const std::exception& e){ print_status("GaussianMutation<Real>", false, e.what()); all_ok=false; }
        x = make_real(10);
        try { UniformMutation op; op.mutate(x, 0.5, lo, hi); print_status("UniformMutation<Real>", x.size()==10); } catch(const std::exception& e){ print_status("UniformMutation<Real>", false, e.what()); all_ok=false; }
    }

    // Mutation binary
    {
        auto b = make_bits(12);
        try { BitFlipMutation op; op.mutate(b, 0.5); print_status("BitFlipMutation<Bit>", b.size()==12); } catch(const std::exception& e){ print_status("BitFlipMutation<Bit>", false, e.what()); all_ok=false; }
    }

    // Mutation integer
    {
        auto xi = make_ints(10);
        try { RandomResettingMutation op; op.mutate(xi, 0.5, -5, 5); print_status("RandomResettingMutation<Int>", xi.size()==10); } catch(const std::exception& e){ print_status("RandomResettingMutation<Int>", false, e.what()); all_ok=false; }
        xi = make_ints(10);
        try { CreepMutation op; op.mutate(xi, 0.5, 1, -10, 10); print_status("CreepMutation<Int>", xi.size()==10); } catch(const std::exception& e){ print_status("CreepMutation<Int>", false, e.what()); all_ok=false; }
    }

    // Mutation permutation
    {
        auto p = make_perm(10);
        try { SwapMutation op; op.mutate(p, 0.8); bool ok = (int)p.size()==10; print_status("SwapMutation<Perm>", ok); if(!ok) all_ok=false; } catch(const std::exception& e){ print_status("SwapMutation<Perm>", false, e.what()); all_ok=false; }
        p = make_perm(10);
        try { InsertMutation op; op.mutate(p, 0.8); bool ok = (int)p.size()==10; print_status("InsertMutation<Perm>", ok); if(!ok) all_ok=false; } catch(const std::exception& e){ print_status("InsertMutation<Perm>", false, e.what()); all_ok=false; }
        p = make_perm(10);
        try { ScrambleMutation op; op.mutate(p, 0.8); bool ok = (int)p.size()==10; print_status("ScrambleMutation<Perm>", ok); if(!ok) all_ok=false; } catch(const std::exception& e){ print_status("ScrambleMutation<Perm>", false, e.what()); all_ok=false; }
        p = make_perm(10);
        try { InversionMutation op; op.mutate(p, 0.8); bool ok = (int)p.size()==10; print_status("InversionMutation<Perm>", ok); if(!ok) all_ok=false; } catch(const std::exception& e){ print_status("InversionMutation<Perm>", false, e.what()); all_ok=false; }
    }

    // List mutation
    {
        std::vector<int> list = {1,2,3,4,5};
        try { ListMutation op; op.mutate(list, 0.5, 0.5, 0, 10, 3, 10); bool ok = list.size()>=3 && list.size()<=10; print_status("ListMutation<List>", ok); if(!ok) all_ok=false; } catch(const std::exception& e){ print_status("ListMutation<List>", false, e.what()); all_ok=false; }
    }

    // Selection operators
    {
        std::vector<Individual> pop;
        for(int i=0;i<20;++i){ Individual ind; ind.fitness = i; ind.genes = std::vector<double>(5, i); pop.push_back(ind);} // increasing fitness
        try { auto idx = TournamentSelection::selectIndices(pop, 3); bool ok = !idx.empty() && idx[0] < pop.size(); print_status("TournamentSelection", ok); if(!ok) all_ok=false; } catch(const std::exception& e){ print_status("TournamentSelection", false, e.what()); all_ok=false; }
        try { auto idx = RouletteWheelSelection::selectIndices(pop, 5); bool ok = !idx.empty() && idx[0] < pop.size(); print_status("RouletteWheelSelection", ok); if(!ok) all_ok=false; } catch(const std::exception& e){ print_status("RouletteWheelSelection", false, e.what()); all_ok=false; }
        try { std::vector<unsigned int> idx; // RankSelection selection API returns vectors of Individuals via select; use select on top
              RankSelection rank; auto sel = rank.select(pop, 5); bool ok = sel.size()==5; print_status("RankSelection", ok); if(!ok) all_ok=false; } catch(const std::exception& e){ print_status("RankSelection", false, e.what()); all_ok=false; }
        try { StochasticUniversalSampling sus; auto sel = sus.select(pop, 5); bool ok = sel.size()==5; print_status("StochasticUniversalSampling", ok); if(!ok) all_ok=false; } catch(const std::exception& e){ print_status("StochasticUniversalSampling", false, e.what()); all_ok=false; }
        try { auto idx = ElitismSelection::selectIndices(pop, 3); bool ok = idx.size()==3; print_status("ElitismSelection", ok); if(!ok) all_ok=false; } catch(const std::exception& e){ print_status("ElitismSelection", false, e.what()); all_ok=false; }
    }

    cout << (all_ok?"ALL PASS":"SOME FAILURES") << endl;
    return all_ok?0:1;
}
