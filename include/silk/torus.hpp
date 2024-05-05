#include <cpads/core.hpp>

namespace kc {

/**
 * In Silk we store a 64x64 grid as a uint4 where the elements
 * encode the four 32x32 quarters of the grid:
 *
 *   +-----+-----+
 *   |     |     |
 *   | a.x | a.y |
 *   |     |     |
 *   +-----+-----+
 *   |     |     |
 *   | a.z | a.w |
 *   |     |     |
 *   +-----+-----+
 *
 * Within each 32x32 square, the kth thread in the warp stores
 * the kth row of the square as a uint32.
 */
_DI_ void shift_torus_inplace(uint32_t &x, uint32_t &y, uint32_t &z, uint32_t &w, int i, int j) {

    // pack elements:
    uint64_t xy = (((uint64_t) y) << 32) | x;
    uint64_t zw = (((uint64_t) w) << 32) | z;

    {
        // vertical shift:
        int src = ((threadIdx.x & 31) - j) & 63;
        uint64_t xy2 = hh::shuffle_32(xy, src);
        uint64_t zw2 = hh::shuffle_32(zw, src);
        xy = (src >= 32) ? zw2 : xy2;
        zw = (src >= 32) ? xy2 : zw2;
    }

    {
        // horizontal shift:
        int ls = i & 63;
        int rs = (-i) & 63;
        xy = (xy << ls) | (xy >> rs);
        zw = (zw << ls) | (zw >> rs);
    }

    // unpack elements:
    x = xy;
    y = xy >> 32;
    z = zw;
    w = zw >> 32;
}

_DI_ uint4 shift_torus(uint4 a, int i, int j) {

    uint4 b = a;
    shift_torus_inplace(b.x, b.y, b.z, b.w, i, j);
    return b;

}

}
