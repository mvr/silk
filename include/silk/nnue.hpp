#pragma once
#include <cpads/core.hpp>

namespace kc {

/**
 * Parametrises the radius-3 closed disc as follows:
 *
 *               1
 *        14 18  5  9 13
 *        10 22 25 21 17
 *      2  6 26 28 24  4  0
 *        19 23 27 20  8
 *        15 11  7 16 12
 *               3
 */
_HD_ void lane2coords(uint32_t lane, int32_t &x, int32_t &y) {

    uint32_t uv = (0x0159a623 >> (lane & 28)) & 15;
    int32_t u = uv & 3;
    int32_t v = uv >> 2;

    if (lane & 2) { u = -u; v = -v; }

    x = (lane & 1) ?   v  : u;
    y = (lane & 1) ? (-u) : v;

}

/**
 * Evaluates the Silk NNUE on a signature of 29 uint9s.
 *
 * The Silk NNUE has 1906816 parameters of which 9984 are active.
 * The parameters are stored contiguously in a GPU-resident array
 * of 7627264 bytes. This function accepts a pointer to that array.
 */
_DI_ float evaluate_nnue(uint32_t signature, const float4 *nnue) {

    // first linear layer: 14848 --> 128
    const float4* nnue_tid = nnue + threadIdx.x;
    uint32_t sparse_row_offset = 16384 * threadIdx.x + 32 * signature + 1568;

    // load contribution from centre cell:
    uint32_t sro_28 = hh::shuffle_32(sparse_row_offset, 28);
    float4 acc = nnue_tid[sro_28];

    // add contributions from other 28 cells:
    #pragma unroll 4
    for (int i = 0; i < 28; i++) {
        uint32_t sro_i = hh::shuffle_32(sparse_row_offset, i);
        float4 row = nnue_tid[sro_i];
        acc.x += row.x;
        acc.y += row.y;
        acc.z += row.z;
        acc.w += row.w;
    }

    // ReLU: 128 --> 128
    acc.x = hh::max(acc.x, 0.0f);
    acc.y = hh::max(acc.y, 0.0f);
    acc.z = hh::max(acc.z, 0.0f);
    acc.w = hh::max(acc.w, 0.0f);

    // load biases:
    float4 biases = nnue_tid[1536];

    // second linear layer: 128 --> 32
    float layer2 = biases.x;
    #pragma unroll 4
    for (int i = 0; i < 32; i++) {
        float4 row = nnue_tid[32 * i];
        float ip = acc.x * row.x + acc.y * row.y + acc.z * row.z + acc.w * row.w;
        layer2 += hh::shuffle_xor_32(ip, i);
    }

    // CReLU: 32 --> 64
    acc.x = hh::max(layer2, 0.0f);
    acc.y = hh::shuffle_xor_32(acc.x, 16);
    acc.z = hh::max(-layer2, 0.0f);
    acc.w = hh::shuffle_xor_32(acc.z, 16);

    // third linear layer: 64 --> 32
    float layer3 = biases.y;
    #pragma unroll 4
    for (int i = 32; i < 48; i++) {
        float4 row = nnue_tid[32 * i];
        float ip = acc.x * row.x + acc.y * row.y + acc.z * row.z + acc.w * row.w;
        layer3 += hh::shuffle_xor_32(ip, i);
    }

    // CReLU: 32 --> 64
    // fourth linear layer: 64 --> 1
    float output = hh::max(layer3, 0.0f) * biases.z + hh::max(-layer3, 0.0f) * biases.w;
    output += hh::shuffle_xor_32(output, 1);
    output += hh::shuffle_xor_32(output, 2);
    output += hh::shuffle_xor_32(output, 4);
    output += hh::shuffle_xor_32(output, 8);
    output += hh::shuffle_xor_32(output, 16);

    return output;
}

} // namespace kc
