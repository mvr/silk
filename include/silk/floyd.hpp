#pragma once
#include "unknown.hpp"

namespace kc {

/**
 * Return codes:
 * -3: reached ambiguity but made progress (so do unit propagation)
 * -2: made no progress (so we need to split into two subproblems)
 * -1: contradiction obtained
 *  0: clean fizzle
 *  1: restablised into different still-life
 *  n: oscillator with period n (n >= 2)
 */
template<bool CollectMetrics, bool HasStator, bool HasExempt>
_DI_ int floyd_cycle(
        uint4 &ad0, uint4 &ad1, uint4 &ad2, uint4 &al2, uint4 &al3, uint4 &ad4, uint4 &ad5, uint4 &ad6, uint4 &stator, uint4 &exempt,
        uint32_t &perturbation, uint32_t &px, uint32_t &py, uint32_t &perturbed_time, uint32_t &restored_time, int max_width, int max_height, int max_pop, int max_perturbed_time, int min_stable, uint32_t* metrics = nullptr
    ) {

    // a half-speed version of perturbation for Floyd's algorithm:
    uint32_t qerturbation = perturbation;
    uint32_t qx = px;
    uint32_t qy = py;

    bool in_cycle = false;
    int generation = 0;

    while (true) {

        {
            // unpack unknown state from stable state and perturbation:
            uint32_t forced_dead = al2.x & al3.x;
            uint32_t forced_live = ad0.x & ad1.x & ad2.x & ad4.x & ad5.x & ad6.x;
            uint32_t not_low = (~perturbation) | forced_dead;
            uint32_t not_high = (~perturbation) | forced_live;
            uint32_t not_stable = perturbation;

            kc::inplace_advance_unknown<false>(ad0.x, ad1.x, ad2.x, al2.x, al3.x, ad4.x, ad5.x, ad6.x, not_low, not_high, not_stable, stator.x, exempt.x, max_width, max_height, max_pop);

            uint32_t contradiction = not_low & not_high & not_stable;

            // printf("Thread %d has contradiction %d\n", threadIdx.x, contradiction);

            if (hh::ballot_32(contradiction)) {
                generation = -1; break; // contradiction obtained
            }

            if (hh::ballot_32(apply_min3(not_low, not_high, not_stable))) {
                generation = (generation == 0) ? -2 : -3; break; // ambiguous
            }

            if (hh::ballot_32(not_stable) == 0) {
                generation = 0; break; // fizzle
            }

            // advance by one generation:
            perturbation = not_stable;
            generation += 1;
            if (hh::ballot_32(perturbation & ad0.x) != 0) {
                perturbed_time++;
                // This also registers that the reaction has started
                // mvrnote: this doesn't detect smothering interactions!
                restored_time = 0;
            } else {
                perturbed_time = 0;
                if (restored_time != ~0)
                    restored_time += 1;
            }
            bump_counter<CollectMetrics>(metrics, METRIC_FLOYD_ITER);
        }

        if (restored_time != ~0 && restored_time >= min_stable) {
            generation = 1000000000;
            break;
        }

        if (perturbed_time >= max_perturbed_time) {
          generation = -1; break; // reaction has gone too long
        }

        {
            // recentre torus on active perturbation:
            uint32_t mx = get_middle_horizontal(perturbation);
            uint32_t my = get_middle_vertical(perturbation);

            shift_torus_inplace(ad0, -mx, -my);
            shift_torus_inplace(ad1, -mx, -my);
            shift_torus_inplace(ad2, -mx, -my);
            shift_torus_inplace(al2, -mx, -my);
            shift_torus_inplace(al3, -mx, -my);
            shift_torus_inplace(ad4, -mx, -my);
            shift_torus_inplace(ad5, -mx, -my);
            shift_torus_inplace(ad6, -mx, -my);
            if constexpr (HasStator) { shift_torus_inplace(stator, -mx, -my); }
            if constexpr (HasExempt) { shift_torus_inplace(exempt, -mx, -my); }
            shift_plane_inplace(perturbation, -mx, -my);

            px = (px + mx) & 63;
            py = (py + my) & 63;
        }

        if ((px == qx) && (py == qy)) {
            if (hh::ballot_32(perturbation != qerturbation) == 0) {
                // a match:
                if (in_cycle) {
                    // we have found the oscillator period:
                    break;
                } else {
                    // the oscillator has entered a cycle:
                    generation = 0;
                    qerturbation = perturbation;
                    qx = px;
                    qy = py;
                    in_cycle = true;
                }
            }
        }

        if (((generation & 1) == 0) && (!in_cycle)) {
            // rotate into reference frame of tortoise:
            uint32_t mx = qx - px;
            uint32_t my = qy - py;

            shift_torus_inplace(ad0, -mx, -my);
            shift_torus_inplace(ad1, -mx, -my);
            shift_torus_inplace(ad2, -mx, -my);
            shift_torus_inplace(al2, -mx, -my);
            shift_torus_inplace(al3, -mx, -my);
            shift_torus_inplace(ad4, -mx, -my);
            shift_torus_inplace(ad5, -mx, -my);
            shift_torus_inplace(ad6, -mx, -my);
            if constexpr (HasStator) { shift_torus_inplace(stator, -mx, -my); }
            if constexpr (HasExempt) { shift_torus_inplace(exempt, -mx, -my); }

            // advance tortoise:
            uint32_t forced_dead = al2.x & al3.x;
            uint32_t forced_live = ad0.x & ad1.x & ad2.x & ad4.x & ad5.x & ad6.x;
            uint32_t not_low = (~qerturbation) | forced_dead;
            uint32_t not_high = (~qerturbation) | forced_live;
            uint32_t not_stable = qerturbation;

            kc::inplace_advance_unknown<false>(ad0.x, ad1.x, ad2.x, al2.x, al3.x, ad4.x, ad5.x, ad6.x, not_low, not_high, not_stable, stator.x, exempt.x, max_width, max_height, max_pop);
            qerturbation = not_stable;
            bump_counter<CollectMetrics>(metrics, METRIC_FLOYD_ITER);

            // rotate back again:
            shift_torus_inplace(ad0, mx, my);
            shift_torus_inplace(ad1, mx, my);
            shift_torus_inplace(ad2, mx, my);
            shift_torus_inplace(al2, mx, my);
            shift_torus_inplace(al3, mx, my);
            shift_torus_inplace(ad4, mx, my);
            shift_torus_inplace(ad5, mx, my);
            shift_torus_inplace(ad6, mx, my);
            if constexpr (HasStator) { shift_torus_inplace(stator, mx, my); }
            if constexpr (HasExempt) { shift_torus_inplace(exempt, mx, my); }

            // recentre tortoise:
            mx = get_middle_horizontal(qerturbation);
            my = get_middle_vertical(qerturbation);
            shift_plane_inplace(qerturbation, -mx, -my);
            qx = (qx + mx) & 63;
            qy = (qy + my) & 63;
        }
    } // while (true)

    if (generation > 0 && min_stable != 999999999 && hh::ballot_32(perturbation & ad0.x) == 0) {
        generation = 1000000000;
    }

    return generation;
}

}
