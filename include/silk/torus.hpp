#include "core.hpp"

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
_DI_ uint4 shift_torus(uint4 a, int i, int j) {

    // pack elements:
    uint64_t xy = (((uint64_t) a.y) << 32) | a.x;
    uint64_t zw = (((uint64_t) a.w) << 32) | a.z;

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

    uint4 b;
    b.x = xy;
    b.y = xy >> 32;
    b.z = zw;
    b.w = zw >> 32;

    return b;
}

}
