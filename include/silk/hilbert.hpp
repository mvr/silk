#pragma once
#include <cpads/core.hpp>

namespace kc {

// Rotate/flip a quadrant appropriately
inline void hilbert_rot(uint32_t n, uint32_t *x, uint32_t *y, uint64_t rx, uint64_t ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }

        // Swap x and y
        uint32_t t  = *x;
        *x = *y;
        *y = t;
    }
}

// Convert a Hilbert curve linear distance p into a (x, y) pair
inline uint64_t hilbert_d2xy(uint64_t p) {

    uint32_t x = 0;
    uint32_t y = 0;
    uint64_t t = p;
    uint32_t s = 1;

    while (t > 0) {
        for (int i = 0; i < 2; i++) {
            uint64_t rx = 1 & (t >> 1);
            uint64_t ry = 1 & (t ^ rx);
            hilbert_rot(s, &x, &y, rx, ry);
            x += s * rx;
            y += s * ry;
            t >>= 2;
            s <<= 1;
        }
    }

    // pack the coordinates together:
    return (((uint64_t) y) << 32) | x;
}

}
