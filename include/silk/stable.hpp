#pragma once
#include "torus.hpp"
#include "metrics.hpp"

namespace kc {

_DI_ uint32_t apply_maj3(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t res; // = (x & y) | (y & z) | (z & x);
    asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(res) : "r"(x), "r"(y), "r"(z));
    return res;
}

_DI_ uint32_t apply_min3(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t res;
    asm("lop3.b32 %0, %1, %2, %3, 0b00010111;" : "=r"(res) : "r"(x), "r"(y), "r"(z));
    return res;
}

_DI_ uint32_t apply_xor3(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t res; // = x ^ y ^ z;
    asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(res) : "r"(x), "r"(y), "r"(z));
    return res;
}

#include "generated/domino.hpp"

/**
 * Determine which cells are fully determined (7 of the 8 bitplanes
 * are set); these are cells for which branching is meaningless.
 */
_DI_ uint32_t is_determined(uint32_t ad0, uint32_t ad1, uint32_t ad2, uint32_t al2, uint32_t al3, uint32_t ad4, uint32_t ad5, uint32_t ad6) {
    uint32_t x2 = apply_maj3(ad0, ad1, ad2);
    uint32_t x1 = apply_xor3(ad0, ad1, ad2);
    uint32_t y2 = apply_maj3(ad4, ad5, ad6);
    uint32_t y1 = apply_xor3(ad4, ad5, ad6);
    uint32_t z2 = apply_maj3(al2, al3, x1);
    uint32_t z1 = apply_xor3(al2, al3, x1);
    return x2 & y2 & z2 & (y1 | z1);
}

template<bool CollectMetrics>
_DI_ bool stableprop(uint32_t &ad0, uint32_t &ad1, uint32_t &ad2, uint32_t &al2, uint32_t &al3, uint32_t &ad4, uint32_t &ad5, uint32_t &ad6, uint32_t* metrics = nullptr) {

    bump_counter<CollectMetrics>(metrics, METRIC_STABLEPROP);

    {
        uint32_t contradiction = ad0 & ad1 & ad2 & al2 & al3 & ad4 & ad5 & ad6;
        contradiction = hh::ballot_32(contradiction != 0);
        if (contradiction) { return true; }
    }

    int no_progress = 0;

    while (no_progress < 2) {

        bump_counter<CollectMetrics>(metrics, METRIC_DOMINO_RULE);
        if (apply_domino_rule<false>(ad0, ad1, ad2, al2, al3, ad4, ad5, ad6)) {
            no_progress = 0;
            uint32_t contradiction = ad0 & ad1 & ad2 & al2 & al3 & ad4 & ad5 & ad6;
            contradiction = hh::ballot_32(contradiction != 0);
            if (contradiction) { return true; }
        } else {
            no_progress += 1;
        }

        if (no_progress >= 2) { break; }

        bump_counter<CollectMetrics>(metrics, METRIC_DOMINO_RULE);
        if (apply_domino_rule<true>(ad0, ad1, ad2, al2, al3, ad4, ad5, ad6)) {
            no_progress = 0;
            uint32_t contradiction = ad0 & ad1 & ad2 & al2 & al3 & ad4 & ad5 & ad6;
            contradiction = hh::ballot_32(contradiction != 0);
            if (contradiction) { return true; }
        } else {
            no_progress += 1;
        }
    }

    return false;
}


/**
 * Apply soft branching on top of an arbitrary inference rule.
 */
template<typename Fn>
_DI_ bool apply_branched(Fn lambda, uint32_t mask, uint32_t *smem, uint32_t &ad0, uint32_t &ad1, uint32_t &ad2, uint32_t &al2, uint32_t &al3, uint32_t &ad4, uint32_t &ad5, uint32_t &ad6) {

    uint32_t last_progress = 0;
    uint32_t p = 0;

    while (p - last_progress < 1024) {

        uint32_t unknowns = mask &~ is_determined(ad0, ad1, ad2, al2, al3, ad4, ad5, ad6);

        p = compute_next_cell(unknowns, p);
        // printf("Thread %d has p = %d\n", threadIdx.x, p);
        if (p == 0xffffffffu) { return false; }

        // push the parent state onto a shared memory 'stack':
        smem[threadIdx.x]       = ad0; ad0 = 0xffffffffu;
        smem[threadIdx.x + 32]  = ad1; ad1 = 0xffffffffu;
        smem[threadIdx.x + 64]  = ad2; ad2 = 0xffffffffu;
        smem[threadIdx.x + 96]  = al2; al2 = 0xffffffffu;
        smem[threadIdx.x + 128] = al3; al3 = 0xffffffffu;
        smem[threadIdx.x + 160] = ad4; ad4 = 0xffffffffu;
        smem[threadIdx.x + 192] = ad5; ad5 = 0xffffffffu;
        smem[threadIdx.x + 224] = ad6; ad6 = 0xffffffffu;

        uint32_t this_cell = (1u << (p & 31));
        this_cell = (threadIdx.x == ((p >> 5) & 31)) ? this_cell : 0u;

        #pragma unroll 1
        for (int i = 0; i < 8; i++) {
            uint32_t r = smem[threadIdx.x + i * 32];
            if (hh::ballot_32(r & this_cell)) {
                // we have ruled out state i already
                continue;
            }
            uint32_t bd0 = smem[threadIdx.x      ] | ((i == 0) ? 0u : this_cell);
            uint32_t bd1 = smem[threadIdx.x + 32 ] | ((i == 1) ? 0u : this_cell);
            uint32_t bd2 = smem[threadIdx.x + 64 ] | ((i == 2) ? 0u : this_cell);
            uint32_t bl2 = smem[threadIdx.x + 96 ] | ((i == 3) ? 0u : this_cell);
            uint32_t bl3 = smem[threadIdx.x + 128] | ((i == 4) ? 0u : this_cell);
            uint32_t bd4 = smem[threadIdx.x + 160] | ((i == 5) ? 0u : this_cell);
            uint32_t bd5 = smem[threadIdx.x + 192] | ((i == 6) ? 0u : this_cell);
            uint32_t bd6 = smem[threadIdx.x + 224] | ((i == 7) ? 0u : this_cell);

            bool contradiction = lambda(bd0, bd1, bd2, bl2, bl3, bd4, bd5, bd6);
            if (contradiction) { continue; }

            ad0 &= bd0;
            ad1 &= bd1;
            ad2 &= bd2;
            al2 &= bl2;
            al3 &= bl3;
            ad4 &= bd4;
            ad5 &= bd5;
            ad6 &= bd6;
        }

        uint32_t contradiction = ad0 & ad1 & ad2 & al2 & al3 & ad4 & ad5 & ad6;
        contradiction = hh::ballot_32(contradiction != 0);
        if (contradiction) { return true; }

        uint32_t progress = 0;
        progress |= (ad0 &~ smem[threadIdx.x]);
        progress |= (ad1 &~ smem[threadIdx.x + 32]);
        progress |= (ad2 &~ smem[threadIdx.x + 64]);
        progress |= (al2 &~ smem[threadIdx.x + 96]);
        progress |= (al3 &~ smem[threadIdx.x + 128]);
        progress |= (ad4 &~ smem[threadIdx.x + 160]);
        progress |= (ad5 &~ smem[threadIdx.x + 192]);
        progress |= (ad6 &~ smem[threadIdx.x + 224]);

        if (hh::ballot_32(progress)) { last_progress = p; }
    }

    return false;

}

/**
 * stableprop with soft branching
 */
_DI_ bool branched_stableprop(uint32_t *smem, uint32_t &ad0, uint32_t &ad1, uint32_t &ad2, uint32_t &al2, uint32_t &al3, uint32_t &ad4, uint32_t &ad5, uint32_t &ad6) {
    return apply_branched([&](uint32_t &bd0, uint32_t &bd1, uint32_t &bd2, uint32_t &bl2, uint32_t &bl3, uint32_t &bd4, uint32_t &bd5, uint32_t &bd6) __attribute__((always_inline)) {
        return stableprop<false>(bd0, bd1, bd2, bl2, bl3, bd4, bd5, bd6);
    }, 0xffffffffu, smem, ad0, ad1, ad2, al2, al3, ad4, ad5, ad6);
}

/**
 * stableprop with (soft branching)^2
 * This is for illustrative purposes only; do not use as it is really slow
 */
_DI_ bool branched_stableprop2(uint32_t *smem, uint32_t &ad0, uint32_t &ad1, uint32_t &ad2, uint32_t &al2, uint32_t &al3, uint32_t &ad4, uint32_t &ad5, uint32_t &ad6) {
    return apply_branched([&](uint32_t &bd0, uint32_t &bd1, uint32_t &bd2, uint32_t &bl2, uint32_t &bl3, uint32_t &bd4, uint32_t &bd5, uint32_t &bd6) __attribute__((always_inline)) {
        return branched_stableprop(smem + 256, bd0, bd1, bd2, bl2, bl3, bd4, bd5, bd6);
    }, 0xffffffffu, smem, ad0, ad1, ad2, al2, al3, ad4, ad5, ad6);
}

} // namespace kc

