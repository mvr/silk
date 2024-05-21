#pragma once
#include "floyd.hpp"
#include "rollout.hpp"
#include "nnue.hpp"

namespace kc {

_DI_ void load4(const uint4* ptr, uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
    uint4 r = ptr[threadIdx.x];
    a = r.x; b = r.y; c = r.z; d = r.w;
}

_DI_ void save4(uint4* ptr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    uint4 r; r.x = a; r.y = b; r.z = c; r.w = d;
    ptr[threadIdx.x] = r;
}

/**
 * Return codes are in the same as kc::floyd_cycle, but this alternates
 * Floyd cycles with branched rollouts until reaching a fixed point.
 */
_DI_ int mainloop(
        uint4 &ad0, uint4 &ad1, uint4 &ad2, uint4 &al2, uint4 &al3, uint4 &ad4, uint4 &ad5, uint4 &ad6, uint4 &stator,
        uint32_t &perturbation, uint32_t &px, uint32_t &py, int max_width, int max_height, int max_pop, int gens,
        uint32_t* smem, uint32_t* metrics
    ) {

    bump_counter(metrics, METRIC_KERNEL);

    int return_code = -3;

    while (return_code == -3) {
        bool contradiction = branched_rollout(smem, ad0.x, ad1.x, ad2.x, al2.x, al3.x, ad4.x, ad5.x, ad6.x, perturbation, stator.x, max_width, max_height, max_pop, gens, metrics);
        if (contradiction) { return_code = -1; break; }
        return_code = floyd_cycle(ad0, ad1, ad2, al2, al3, ad4, ad5, ad6, stator, perturbation, px, py, max_width, max_height, max_pop, metrics);
    }

    return return_code;
}

template<typename Fn>
_DI_ float hard_branch(
        uint4* writing_location, uint32_t perturbation, uint32_t &metadata_z,
        uint32_t &ad0, uint32_t &ad1, uint32_t &ad2, uint32_t &al2, uint32_t &al3, uint32_t &ad4, uint32_t &ad5, uint32_t &ad6,
        uint32_t stator, int max_width, int max_height, int max_pop, uint32_t *smem, float epsilon, Fn lambda
    ) {

    uint32_t not_stable = perturbation;
    uint32_t forced_dead = al2 & al3;
    uint32_t forced_live = ad0 & ad1 & ad2 & ad4 & ad5 & ad6;
    uint32_t not_low = (~not_stable) | forced_dead;
    uint32_t not_high = (~not_stable) | forced_live;

    kc::inplace_advance_unknown(ad0, ad1, ad2, al2, al3, ad4, ad5, ad6, not_low, not_high, not_stable, stator, max_width, max_height, max_pop, smem);

    uint32_t ambiguous = apply_min3(not_low, not_high, not_stable);

    uint32_t unknown_if_stable = ambiguous &~ not_stable;

    uint32_t p = compute_next_cell(ambiguous, 0);
    uint32_t best_p = p;
    float best_loss = 1.0e30f;

    int32_t dx = 0;
    int32_t dy = 0;
    lane2coords(threadIdx.x, dx, dy);

    uint32_t random_a, random_b;
    {
        // hash the contents inside this thread:
        uint64_t randomness = threadIdx.x;
        randomness = randomness * 0x7dcf411d + ad0;
        randomness = randomness * 0x7dcf411d + ad1;
        randomness = randomness * 0x7dcf411d + ad2;
        randomness = randomness * 0x7dcf411d + al2;
        randomness = randomness * 0x7dcf411d + al3;
        randomness = randomness * 0x7dcf411d + ad4;
        randomness = randomness * 0x7dcf411d + ad5;
        randomness = randomness * 0x7dcf411d + ad6;

        // apply mixing function:
        randomness = hh::fibmix(randomness);
        randomness = hh::fibmix(randomness);

        // split into high and low components:
        random_a = randomness;
        random_b = randomness >> 32;

        // ensure all cells get consistent randomness in [0, 4194304]:
        random_a = hh::warp_add(random_a) >> 10;
        random_b = hh::warp_add(random_b) >> 10;
    }

    uint32_t epsilon_threshold = ((uint32_t) (epsilon * 4194304.0f));

    // In the epsilon-greedy strategy for multi-armed bandit, we sometimes
    // (with probability epsilon) decide to choose a random lever instead
    // of the best one:
    bool choose_random_lever = (random_a < epsilon_threshold);

    {
        // Generate a random number in range(population):
        uint32_t population = hh::warp_add((uint32_t) hh::popc32(ambiguous));
        random_b *= population;
        random_b = random_b >> 22;
    }

    uint32_t cell_idx = 0;
    while (p < 1024) {

        // only do the evaluation if we need to:
        if ((cell_idx == random_b) || (!choose_random_lever)) {

            uint32_t ex = (dx + p) & 31;
            uint32_t ey = (dy + (p >> 5)) & 31;

            // compute a signature (29 values in the interval [0, 511]) that
            // we evaluate with our custom function lambda:
            uint32_t signature = 0;
            signature += ((hh::shuffle_32(ad0, ey) >> ex) & 1);
            signature += ((hh::shuffle_32(ad1, ey) >> ex) & 1) * 2;
            signature += ((hh::shuffle_32(ad2, ey) >> ex) & 1) * 4;
            signature += ((hh::shuffle_32(al2, ey) >> ex) & 1) * 8;
            signature += ((hh::shuffle_32(al3, ey) >> ex) & 1) * 16;
            signature += ((hh::shuffle_32(ad4, ey) >> ex) & 1) * 32;
            signature += ((hh::shuffle_32(ad5, ey) >> ex) & 1) * 64;
            signature += ((hh::shuffle_32(ad6, ey) >> ex) & 1) * 128;
            signature += ((hh::shuffle_32(perturbation, ey) >> ex) & 1) * 256;

            // evaluate heuristic:
            float loss = lambda(signature);

            if ((loss < best_loss) || choose_random_lever) {
                // select this lever:
                best_loss = loss;
                best_p = p;
                metadata_z = signature;
            }
        }

        p = compute_next_cell(ambiguous, p);
        cell_idx += 1;
    }

    uint32_t this_cell = (1u << (best_p & 31));
    this_cell = (threadIdx.x == ((best_p >> 5) & 31)) ? this_cell : 0u;

    // if a cell is unknown as to whether it's stable, then we
    // copy from smem; otherwise, we set to 0 for bd and 1 for bl:
    uint32_t bd0 = unknown_if_stable & smem[threadIdx.x];
    uint32_t bd1 = unknown_if_stable & smem[threadIdx.x + 32];
    uint32_t bd2 = unknown_if_stable & smem[threadIdx.x + 64];
    uint32_t bl2 = (~unknown_if_stable) | smem[threadIdx.x + 96];
    uint32_t bl3 = (~unknown_if_stable) | smem[threadIdx.x + 128];
    uint32_t bd4 = unknown_if_stable & smem[threadIdx.x + 160];
    uint32_t bd5 = unknown_if_stable & smem[threadIdx.x + 192];
    uint32_t bd6 = unknown_if_stable & smem[threadIdx.x + 224];

    kc::save4(
        writing_location + 192,
        ad0 | (this_cell & bd0),
        ad1 | (this_cell & bd1),
        ad2 | (this_cell & bd2),
        al2 | (this_cell & bl2)
    );
    kc::save4(
        writing_location + 224,
        al3 | (this_cell & bl3),
        ad4 | (this_cell & bd4),
        ad5 | (this_cell & bd5),
        ad6 | (this_cell & bd6)
    );
    kc::save4(
        writing_location + 256,
        ad0 | (this_cell &~ bd0),
        ad1 | (this_cell &~ bd1),
        ad2 | (this_cell &~ bd2),
        al2 | (this_cell &~ bl2)
    );
    kc::save4(
        writing_location + 288,
        al3 | (this_cell &~ bl3),
        ad4 | (this_cell &~ bd4),
        ad5 | (this_cell &~ bd5),
        ad6 | (this_cell &~ bd6)
    );

    return best_loss;
}

} // namespace kc

