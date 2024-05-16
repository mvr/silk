#pragma once

#include <silk/completestill.hpp>
#include <cpads/random/prng.hpp>
#include <gtest/gtest.h>

inline void sl_to_bitplanes(const std::vector<uint32_t> &solution, uint32_t* bp) {

    for (int y = 0; y < 256; y++) {
        bp[y] = ((uint32_t) -1);
    }

    //                      d0    d1    d2 l2    l3 d4    d5    d6
    constexpr int arr[18] = {0, 8, 1, 8, 2, 3, 8, 4, 5, 8, 6, 8, 7, 8, 8, 8, 8, 8};

    // just for testing so no need to bitslice this:
    for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 32; x++) {
            int ncount = 0;
            for (int y2 = y - 1; y2 <= y + 1; y2++) {
                for (int x2 = x - 1; x2 <= x + 1; x2++) {
                    ncount += (((x2 == x) && (y2 == y)) ? 1 : 2) * ((solution[y2 & 31] >> (x2 & 31)) & 1);
                }
            }
            int newstate = arr[ncount];

            EXPECT_LT(newstate, 8); // check that we have a still-life

            if (newstate < 8) {
                bp[y + newstate * 32] ^= (1 << x);
            }
        }
    }
}

inline std::vector<uint32_t> make_random_sl(uint64_t seed, uint32_t* ground_truth = nullptr, uint32_t* problem = nullptr, double prob = 0.0) {

    std::vector<uint32_t> known_live(32);
    std::vector<uint32_t> known_dead(32);

    hh::PRNG pcg(seed, seed, seed);

    uint64_t threshold = ((uint64_t) (prob * 0x100000000ull));

    for (int k = 0; k < 32; k++) {
        // width-2 border cells must be dead:
        known_dead[k] = (((k >= 2) && (k < 30)) ? 0xc0000003u : 0xffffffffu);
    }

    for (int y = 1; y < 10; y++) {
        for (int x = 1; x < 10; x++) {
            uint32_t r = pcg.generate();

            int dy = 3 * y + (r & 1);
            int dx = 3 * x + ((r >> 1) & 1);

            known_live[dy] |= (1ull << dx);
        }
    }

    auto constraints = kc::expand_constraints<uint32_t>(&(known_live[0]), &(known_dead[0]));
    EXPECT_EQ(constraints.size(), 256);

    for (int k = 32; k < 256; k++) {
        // width-1 border cells must be dead with 0 live neighbours:
        constraints[k] |= ((((k + 1) & 31) < 2) ? 0xffffffffu : 0x80000001u);
    }

    auto solution = kc::complete_still_life<uint32_t>(&(constraints[0]));
    EXPECT_EQ(solution.size(), 32);

    if (ground_truth != nullptr) {
        sl_to_bitplanes(solution, ground_truth);

        if (problem != nullptr) {
            for (int i = 0; i < 256; i++) {
                uint32_t gt = ground_truth[i];
                uint32_t p = ((((i + 1) & 31) < 2) ? 0xffffffffu : 0x80000001u) & constraints[i];
                for (int j = 0; j < 32; j++) {
                    uint32_t r = pcg.generate();
                    if (r < threshold) { p |= gt & (1u << j); }
                }
                problem[i] = p;
            }
        }
    }

    return solution;
}
