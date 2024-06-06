#pragma once
#include <cpads/core.hpp>

namespace kc {

/**
 * modulus is assumed to be a power of 2
 */
inline uint64_t random_perm(uint64_t input, uint64_t modulus, uint64_t key, int rounds = 6) {

    uint64_t x = input;
    uint64_t k = key;

    for (int i = 0; i < rounds; i++) {
        k = hh::fibmix(k) + 3511;
        x += k;
        x *= (x + x + 1093);
        x &= (modulus - 1);
        x ^= (x >> 11);
    }

    return x;
}

} // namespace kc
