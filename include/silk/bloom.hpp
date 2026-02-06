#pragma once
#include "torus.hpp"
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

_DI_ void warp_bit_permute(uint32_t &x, uint32_t input) {
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
}

/**
 * Produces a 1024-bit value (32 bits per thread) where exactly k of the
 * 1024 bits are set (where k can be between 0 and 32, inclusive). This
 * is a deterministic function of the 2048-bit hash (64 bits per thread).
 */
_DI_ uint32_t sparse_random(uint64_t hash, int k) {
    uint32_t x = 0;
    if ((threadIdx.x & 31) < k) { x = (1u << (hash & 31)); }
    warp_bit_permute(x, ((uint32_t) (hash >> 32)));
    warp_bit_permute(x, ((uint32_t) hash));
    return x;
}

/**
 * Radius-2 neighbourhood with the four corners removed (21 cells).
 */
_DI_ uint32_t zoi21(uint32_t perturbation) {
    uint32_t hm2 = kc::shift_plane<false, -2>(perturbation);
    uint32_t hm1 = kc::shift_plane<false, -1>(perturbation);
    uint32_t hp1 = kc::shift_plane<false,  1>(perturbation);
    uint32_t hp2 = kc::shift_plane<false,  2>(perturbation);

    uint32_t near = hm2 | hm1 | perturbation | hp1 | hp2;
    uint32_t far = hm1 | perturbation | hp1;

    return near
        | kc::shift_plane<true,  1>(near)
        | kc::shift_plane<true, -1>(near)
        | kc::shift_plane<true,  2>(far)
        | kc::shift_plane<true, -2>(far);
}

/**
 * Hash perturbation together with masked stable information in zoi21.
 */
_DI_ uint64_t dedup_hash(
        uint32_t perturbation,
        uint32_t ad0, uint32_t ad1, uint32_t ad2, uint32_t al2,
        uint32_t al3, uint32_t ad4, uint32_t ad5, uint32_t ad6
    ) {

    uint32_t mask = zoi21(perturbation);
    uint64_t x = (((uint64_t) perturbation) << 32) | mask;

    uint64_t p01 = (((uint64_t) (mask & ad0)) << 32) | (mask & ad1);
    uint64_t p23 = (((uint64_t) (mask & ad2)) << 32) | (mask & al2);
    uint64_t p45 = (((uint64_t) (mask & al3)) << 32) | (mask & ad4);
    uint64_t p67 = (((uint64_t) (mask & ad5)) << 32) | (mask & ad6);

    x = hh::fibmix(x ^ 0x8577d6d46a5f60c3ull);
    x = hh::fibmix(x + p01 + 0x243f6a8885a308d3ull);
    x = hh::fibmix(x + p23 + 0x13198a2e03707344ull);
    x = hh::fibmix(x + p45 + 0xa4093822299f31d0ull);
    x = hh::fibmix(x + p67 + 0x082efa98ec4e6c89ull);

    return warp_hash(x);
}

/**
 * Returns true if the hash was already present in the filter.
 */
_DI_ bool bloom_test_and_set(uint32_t* bloom_filter, uint32_t bloom_chunk_mask, uint64_t hash) {
    uint32_t sparse = sparse_random(hash, 32);
    uint64_t lane0_hash = hh::shuffle_32(hash, 0);
    uint32_t chunk = ((uint32_t) (lane0_hash >> 32)) & bloom_chunk_mask;
    uint32_t index = (chunk << 5) + (threadIdx.x & 31);
    uint32_t prev = hh::atomic_or(bloom_filter + index, sparse);
    return (hh::ballot_32((sparse &~ prev) != 0) == 0);
}

}
