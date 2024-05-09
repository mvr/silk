#include <silk/logic.hpp>
#include <silk/completestill.hpp>
#include <cpads/random/prng.hpp>
#include <gtest/gtest.h>

__global__ void stableprop_kernel(uint32_t *x) {

    uint32_t offset = blockIdx.x * 256 + threadIdx.x;

    uint32_t d0 = x[offset];
    uint32_t d1 = x[offset + 32];
    uint32_t d2 = x[offset + 64];
    uint32_t l2 = x[offset + 96];
    uint32_t l3 = x[offset + 128];
    uint32_t d4 = x[offset + 160];
    uint32_t d5 = x[offset + 192];
    uint32_t d6 = x[offset + 224];

    kc::stableprop(d0, d1, d2, l2, l3, d4, d5, d6);

    x[offset] = d0;
    x[offset + 32] = d1;
    x[offset + 64] = d2;
    x[offset + 96] = l2;
    x[offset + 128] = l3;
    x[offset + 160] = d4;
    x[offset + 192] = d5;
    x[offset + 224] = d6;
}

std::vector<uint32_t> make_random_sl(uint64_t seed) {

    std::vector<uint32_t> known_live(32);
    std::vector<uint32_t> known_dead(32);

    hh::PRNG pcg(seed, seed, seed);

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

    return solution;
}


TEST(CompleteStill, Random) {

    for (int i = 0; i < 100; i++) {
        auto solution = make_random_sl(i);
        /*
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                std::cout << (((solution[y] >> x) & 1) ? '*' : '.');
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        */
    }

}
