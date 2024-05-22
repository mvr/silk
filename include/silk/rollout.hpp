#pragma once
#include "unknown.hpp"

namespace kc {

template<bool CollectMetrics>
_DI_ bool run_rollout(
    uint32_t &ad0, uint32_t &ad1, uint32_t &ad2, uint32_t &al2, uint32_t &al3, uint32_t &ad4, uint32_t &ad5, uint32_t &ad6,
    uint32_t perturbation, uint32_t stator, int max_width, int max_height, int max_pop, int gens, uint32_t* metrics = nullptr
) {

    bool improved = true;

    bump_counter<CollectMetrics>(metrics, METRIC_ROLLOUT);

    while (improved) {

        {
            // apply stable propagation rules:
            bool contradiction = kc::stableprop<CollectMetrics>(ad0, ad1, ad2, al2, al3, ad4, ad5, ad6, metrics);
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
        }
    }

    // we reached the end of the rollout with no further deductions:
    return false;
}


template<bool CollectMetrics>
_DI_ bool branched_rollout(
    uint32_t *smem, uint32_t &ad0, uint32_t &ad1, uint32_t &ad2, uint32_t &al2, uint32_t &al3, uint32_t &ad4, uint32_t &ad5, uint32_t &ad6,
    uint32_t perturbation, uint32_t stator, int max_width, int max_height, int max_pop, int gens, uint32_t *metrics = nullptr
) {

    bump_counter<CollectMetrics>(metrics, METRIC_BRANCHING);

    // obtain the 21-cell ZOI of perturbation:
    uint32_t mask = perturbation;
    mask = expand_plane<true, true>(mask);
    mask = expand_plane<true, false>(mask);
    mask = expand_plane<false, true>(mask);

    return apply_branched([&](uint32_t &bd0, uint32_t &bd1, uint32_t &bd2, uint32_t &bl2, uint32_t &bl3, uint32_t &bd4, uint32_t &bd5, uint32_t &bd6) __attribute__((always_inline)) {
        return run_rollout<CollectMetrics>(bd0, bd1, bd2, bl2, bl3, bd4, bd5, bd6, perturbation, stator, max_width, max_height, max_pop, gens, metrics);
    }, mask, smem, ad0, ad1, ad2, al2, al3, ad4, ad5, ad6);
}

}
