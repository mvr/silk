#pragma once
#include "floyd.hpp"
#include "rollout.hpp"
#include "nnue.hpp"

namespace kc {

template<typename Fn>
_DI_ float hard_branch(
        uint4* writing_location, uint32_t perturbation, uint32_t &metadata_z, uint32_t &metadata_out,
        uint32_t &ad0, uint32_t &ad1, uint32_t &ad2, uint32_t &al2, uint32_t &al3, uint32_t &ad4, uint32_t &ad5, uint32_t &ad6,
        uint32_t stator, uint32_t exempt, int max_width, int max_height, int max_pop, uint32_t *smem, uint32_t epsilon_threshold, Fn lambda, uint32_t *metrics
    ) {

    bump_counter<true>(metrics, METRIC_HARDBRANCH);

    // unpack unknown state from stable state and perturbation:
    uint32_t forced_dead = al2 & al3;
    uint32_t forced_live = ad0 & ad1 & ad2 & ad4 & ad5 & ad6;
    uint32_t not_low = (~perturbation) | forced_dead;
    uint32_t not_high = (~perturbation) | forced_live;
    uint32_t not_stable = perturbation;

    uint32_t ambiguous = get_branching_cells<true>(ad0, ad1, ad2, al2, al3, ad4, ad5, ad6, not_low, not_high, not_stable, stator, exempt, max_width, max_height, max_pop, smem);
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

    // In the epsilon-greedy strategy for multi-armed bandit, we sometimes
    // (with probability epsilon) decide to choose a random lever instead
    // of the best one:
    bool choose_random_lever = (random_a < epsilon_threshold);

    {
        // Generate a random number in range(population):
        uint32_t population = hh::warp_add((uint32_t) hh::popc32(ambiguous));
        random_b *= population;
        random_b = random_b >> 22;

        bump_counter<true>(metrics, choose_random_lever ? METRIC_EXPLORE : METRIC_EXPLOIT);
    }

    uint32_t cell_idx = 0;
    while (p < 1024) {

        // only do the evaluation if we need to:
        if ((cell_idx == random_b) || (!choose_random_lever)) {

            bump_counter<true>(metrics, METRIC_NNUE);

            uint32_t ex = (dx + p) & 31;
            uint32_t ey = (dy + (p >> 5)) & 31;

            // compute a signature (29 values in the interval [0, 255]) that
            // we evaluate with our custom function lambda:
            uint32_t signature = 0;

            // incorporate stable state:
            signature += ((hh::shuffle_32(ad0, ey) >> ex) & 1);
            signature += ((hh::shuffle_32(ad1, ey) >> ex) & 1) * 2;
            signature += ((hh::shuffle_32(ad2, ey) >> ex) & 1) * 4;
            signature += ((hh::shuffle_32(al2, ey) >> ex) & 1) * 8;
            signature += ((hh::shuffle_32(al3, ey) >> ex) & 1) * 16;
            signature += ((hh::shuffle_32(ad4, ey) >> ex) & 1) * 32;
            signature += ((hh::shuffle_32(ad5, ey) >> ex) & 1) * 64;
            signature += ((hh::shuffle_32(ad6, ey) >> ex) & 1) * 128;

            // incorporate perturbation:
            bool active = ((hh::shuffle_32(perturbation, ey) >> ex) & 1);
            if (active) {
                uint32_t syndrome = 0xff ^ signature;
                if (syndrome & (syndrome - 1)) {
                    signature = 0xff;
                } else {
                    uint32_t magic = syndrome * syndrome * 0x7dcade;
                    signature = (magic >> 16) & 0x66;
                }
            }

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

    best_p &= 1023;

    // if a cell is unknown as to whether it's stable, then we
    // copy from smem; otherwise, we set to 0 for bd and 1 for bl:
    smem[threadIdx.x]       &=   unknown_if_stable;  // bd0
    smem[threadIdx.x + 32]  &=   unknown_if_stable;  // bd1
    smem[threadIdx.x + 64]  &=   unknown_if_stable;  // bd2
    smem[threadIdx.x + 96]  |= (~unknown_if_stable); // bl2
    smem[threadIdx.x + 128] |= (~unknown_if_stable); // bl3
    smem[threadIdx.x + 160] &=   unknown_if_stable;  // bd4
    smem[threadIdx.x + 192] &=   unknown_if_stable;  // bd5
    smem[threadIdx.x + 224] &=   unknown_if_stable;  // bd6

    __syncthreads();
    int lane8 = threadIdx.x & 7;
    uint32_t contrib = smem[(best_p >> 5) + 32 * lane8];
    __syncthreads();

    contrib >>= (best_p & 31);
    contrib &= 1;
    contrib <<= lane8;
    contrib += hh::shuffle_xor_32(contrib, 1);
    contrib += hh::shuffle_xor_32(contrib, 2);
    contrib += hh::shuffle_xor_32(contrib, 4);

    if (threadIdx.x == 31) { metadata_out = contrib; }
    if (threadIdx.x == 30) { metadata_out = best_p; }

    return best_loss;
}

} // namespace kc

