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

_DI_ float evaluate_nnue(uint32_t signature, const float4 *nnue) {

    // first linear layer: 14848 --> 128
    float4 acc; acc.x = 0.0f; acc.y = 0.0f; acc.z = 0.0f; acc.w = 0.0f;
    for (int i = 0; i < 29; i++) {
        uint32_t sig_i = hh::shuffle_32(signature, i);
        float4 row = nnue[16384 * i + 32 * sig_i + threadIdx.x];
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
    float4 biases = nnue[476672 + threadIdx.x];

    // second linear layer: 128 --> 32
    float layer2 = biases.x;
    for (int i = 0; i < 32; i++) {
        float4 row = nnue[475136 + 32 * i + threadIdx.x];
        float ip = acc.x * row.x + acc.y * row.y + acc.z * row.z + acc.w * row.w;
        layer2 += hh::shuffle_xor_32(ip, i);
    }

    // CReLU: 32 --> 64
    acc.x = hh::max(layer2, 0.0f);
    acc.y = hh::max(-layer2, 0.0f);
    acc.z = hh::shuffle_xor_32(acc.x, 16);
    acc.w = hh::shuffle_xor_32(acc.y, 16);

    // third linear layer: 64 --> 32
    float layer3 = biases.y;
    for (int i = 0; i < 16; i++) {
        float4 row = nnue[476160 + 32 * i + threadIdx.x];
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
