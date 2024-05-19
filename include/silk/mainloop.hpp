#pragma once
#include "floyd.hpp"
#include "rollout.hpp"

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


} // namespace kc

