#pragma once
#include <cpads/core.hpp>

namespace kc {

_DI_ void inplace_vertical_circulant(uint64_t &x, int i, int j) {
    uint32_t y = ((uint32_t) x);
    y = hh::shuffle_32(y, threadIdx.x + i) ^ hh::shuffle_32(y, threadIdx.x + j);
    x ^= y;
    x = hh::fibmix(x);
}

_DI_ uint64_t warp_hash(uint64_t input) {
    uint64_t x = hh::fibmix(input + 6700417 * (threadIdx.x & 31));
    inplace_vertical_circulant(x, 22, 27);
    inplace_vertical_circulant(x, 21, 6);
    inplace_vertical_circulant(x, 18, 9);
    return x;
}

_DI_ uint32_t partial_shuffle(uint32_t x, uint32_t mask, int lane) {
    uint32_t y = hh::shuffle_xor_32(x, lane);
    return (x & mask) | (y &~ mask);
}

_DI_ uint32_t random_rotate(uint32_t x, uint32_t y) {
    return (x << (y & 31)) | (x >> ((-y) & 31));
}

/**
 * Produces a 1024-bit value (32 bits per thread) where exactly k of the
 * 1024 bits are set.
 */
_DI_ uint32_t sparse_random(uint32_t input, int k) {

    uint32_t x = 0;

    if ((threadIdx.x & 31) < k) { x = (1u << (input & 31)); }
    x = partial_shuffle(x, hh::ballot_32(input & 32), 16);
    x = random_rotate(x, input >> 10);
    x = partial_shuffle(x, hh::ballot_32(input & 64), 8);
    x = random_rotate(x, input >> 15);
    x = partial_shuffle(x, hh::ballot_32(input & 128), 4);
    x = random_rotate(x, input >> 20);
    x = partial_shuffle(x, hh::ballot_32(input & 256), 2);
    x = random_rotate(x, input >> 25);
    x = partial_shuffle(x, hh::ballot_32(input & 512), 1);
    x = random_rotate(x, input);
    return x;
}

}
