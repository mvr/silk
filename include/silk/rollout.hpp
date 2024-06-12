#pragma once
#include "unknown.hpp"

namespace kc {

template<bool CollectMetrics>
_DI_ bool run_rollout(
    uint32_t &ad0, uint32_t &ad1, uint32_t &ad2, uint32_t &al2, uint32_t &al3, uint32_t &ad4, uint32_t &ad5, uint32_t &ad6,
    uint32_t perturbation, uint32_t stator, int max_width, int max_height, int max_pop, int gens, uint32_t* metrics = nullptr, int max_rounds = 32768
) {

    bool improved = true;

    bump_counter<CollectMetrics>(metrics, METRIC_ROLLOUT);

    while (improved) {

        {
            // apply stable propagation rules:
            bool contradiction = kc::stableprop<CollectMetrics>(ad0, ad1, ad2, al2, al3, ad4, ad5, ad6, metrics, max_rounds);
            if (contradiction) { return true; }
        }

        improved = false;

        // unpack unknown state from stable state and perturbation:
        uint32_t forced_dead = al2 & al3;
        uint32_t forced_live = ad0 & ad1 & ad2 & ad4 & ad5 & ad6;
        uint32_t not_low = (~perturbation) | forced_dead;
        uint32_t not_high = (~perturbation) | forced_live;
        uint32_t not_stable = perturbation;

        // run rollout:
        for (int i = 0; i < gens; i++) {
            bump_counter<CollectMetrics>(metrics, METRIC_ADVANCE);
            improved = kc::inplace_advance_unknown<false>(ad0, ad1, ad2, al2, al3, ad4, ad5, ad6, not_low, not_high, not_stable, stator, max_width, max_height, max_pop);
            bool contradiction = hh::ballot_32(not_low & not_high & not_stable);
            if (contradiction) { return true; }
            if (improved) { break; }
            if (hh::ballot_32(not_stable) == 0) { break; }
        }
    }

    // we reached the end of the rollout with no further deductions:
    return false;
}

template<bool CopyToSmem>
_DI_ uint32_t get_branching_cells(
    uint32_t &ad0, uint32_t &ad1, uint32_t &ad2, uint32_t &al2, uint32_t &al3, uint32_t &ad4, uint32_t &ad5, uint32_t &ad6,
    uint32_t &not_low, uint32_t &not_high, uint32_t &not_stable,
    uint32_t stator, int max_width = 28, int max_height = 28, uint32_t max_pop = 784, uint32_t* smem = nullptr
) {

    kc::inplace_advance_unknown<CopyToSmem>(ad0, ad1, ad2, al2, al3, ad4, ad5, ad6, not_low, not_high, not_stable, stator, max_width, max_height, max_pop, smem);

    uint32_t ambiguous = apply_min3(not_low, not_high, not_stable);

    {
        uint32_t x2 = apply_maj3(ad1, ad2, al2);
        uint32_t x1 = apply_xor3(ad1, ad2, al2);
        uint32_t y2 = apply_maj3(ad4, ad5, al3);
        uint32_t y1 = apply_xor3(ad4, ad5, al3);
        uint32_t z2 = apply_maj3(x1, y1, ad6);
        uint32_t z1 = apply_xor3(x1, y1, ad6);
        uint32_t q4 = apply_maj3(x2, y2, z2);
        uint32_t w2 = apply_xor3(x2, y2, z2);

        uint32_t q1 =  z1 ^ ad0;
        uint32_t q2 = (z1 & ad0) ^ w2;
        q4 |= (w2 &~ q2);

        // binary search to find the cells with most information:
        if (hh::ballot_32(q4 & ambiguous)) { ambiguous &= q4; }
        if (hh::ballot_32(q2 & ambiguous)) { ambiguous &= q2; }
        if (hh::ballot_32(q1 & ambiguous)) { ambiguous &= q1; }
    }

    return ambiguous;
}

template<bool CollectMetrics>
_DI_ bool branched_rollout(
    uint32_t *smem, uint32_t &ad0, uint32_t &ad1, uint32_t &ad2, uint32_t &al2, uint32_t &al3, uint32_t &ad4, uint32_t &ad5, uint32_t &ad6,
    uint32_t perturbation, uint32_t stator, int max_width, int max_height, int max_pop, int gens, uint32_t *metrics = nullptr, int max_rounds = 32768
) {

    bump_counter<CollectMetrics>(metrics, METRIC_BRANCHING);

    uint32_t cumulative_mask = 0;

    while (true) {

        // unpack unknown state from stable state and perturbation:
        uint32_t forced_dead = al2 & al3;
        uint32_t forced_live = ad0 & ad1 & ad2 & ad4 & ad5 & ad6;
        uint32_t not_low = (~perturbation) | forced_dead;
        uint32_t not_high = (~perturbation) | forced_live;
        uint32_t not_stable = perturbation;

        uint32_t mask = get_branching_cells<false>(ad0, ad1, ad2, al2, al3, ad4, ad5, ad6, not_low, not_high, not_stable, stator, max_width, max_height, max_pop);

        if (hh::ballot_32((mask &~ cumulative_mask) != 0) == 0) { return false; }
        cumulative_mask |= mask;

        bool contradiction = apply_branched<false>([&](uint32_t &bd0, uint32_t &bd1, uint32_t &bd2, uint32_t &bl2, uint32_t &bl3, uint32_t &bd4, uint32_t &bd5, uint32_t &bd6) __attribute__((always_inline)) {
            return run_rollout<CollectMetrics>(bd0, bd1, bd2, bl2, bl3, bd4, bd5, bd6, perturbation, stator, max_width, max_height, max_pop, gens, metrics, max_rounds);
        }, mask, smem, ad0, ad1, ad2, al2, al3, ad4, ad5, ad6);

        if (contradiction) { return true; }
    }
}

}
