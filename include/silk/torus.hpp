#pragma once
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

_DI_ void shift_torus_inplace(uint4 &b, int i, int j) {
    shift_torus_inplace(b.x, b.y, b.z, b.w, i, j);
}

_DI_ void shift_plane_inplace(uint32_t &x, int i, int j) {
    {
        // vertical shift:
        int src = ((threadIdx.x & 31) - j) & 63;
        uint32_t x2 = hh::shuffle_32(x, src);
        x2 = (src >= 32) ? 0 : x2;
        // horizontal shift:
        int ls = i & 63;
        int rs = (-i) & 63;
        x = 0;
        if (ls < 32) { x |= (x2 << ls); }
        if (rs < 32) { x |= (x2 >> rs); }
    }
}

_DI_ uint4 shift_torus(uint4 a, int i, int j) {
    uint4 b = a;
    shift_torus_inplace(b, i, j);
    return b;
}

template<bool direction, int amount, bool negated = false>
_DI_ uint32_t shift_plane(uint32_t x) {

    constexpr uint32_t smask = negated ? 0xffffffffu : 0u;

    // handle degenerate cases:
    if constexpr (amount == 0) { return x; }
    if constexpr ((amount >= 32) || (amount <= -32)) { return 0; }

    if constexpr (direction) {
        // vertical
        int lid = threadIdx.x & 31;
        if constexpr (amount > 0) {
            uint32_t y = hh::shuffle_up_32(x, amount);
            return (lid >= amount) ? y : smask;
        } else {
            uint32_t y = hh::shuffle_down_32(x, -amount);
            return (lid < 32 + amount) ? y : smask;
        }
    } else {
        // horizontal
        uint32_t y = x;
        if constexpr (smask) { y = ~y; }
        if constexpr (amount > 0) {
            y = (y << amount);
        } else {
            y = (y >> (-amount));
        }
        if constexpr (smask) { y = ~y; }
        return y;
    }
}

_HD_ uint32_t get_middle(uint32_t x) {
    uint32_t bias = hh::ffs32(x) - hh::clz32(x);
    return (bias >> 1);
}

_DI_ uint32_t get_middle_horizontal(uint32_t x) {
    uint32_t y = hh::warp_or(x);
    return get_middle(y);
}

_DI_ uint32_t get_middle_vertical(uint32_t x) {
    uint32_t y = hh::ballot_32(x != 0);
    return get_middle(y);
}

_DI_ uint32_t active_to_inactive(uint32_t x, int p) {
    uint32_t y = x | (-x);
    return (y << p);
}

_DI_ uint32_t get_border() {
    // everything outside central 28x28 square must be stable:
    uint32_t border = (((threadIdx.x + 2) & 31) < 4) ? 0xffffffffu : 0xc0000003u;
    return border;
}

/**
 * Determine the cells that are forced to be stable based on those
 * that are known to be unstable, making use of the width, height,
 * and population bounds of the active region.
 */
_DI_ uint32_t get_forced_stable(uint32_t not_stable, uint32_t ad0, uint32_t stator, uint32_t exempt, int max_width, int max_height, uint32_t max_pop) {

    // the active region consists of cells adjacent to the catalyst
    // that are not in the stable state:
    uint32_t active = not_stable & ad0 & ~exempt;
    uint32_t active_y = hh::ballot_32(active != 0);
    uint32_t active_p = hh::warp_add((uint32_t) hh::popc32(active));
    uint32_t active_x = hh::warp_or(active);

    // determine cells that are too far from the most distant active cell:
    uint32_t inactive_y = active_to_inactive(active_y, max_height);
    inactive_y |= hh::brev32(active_to_inactive(hh::brev32(active_y), max_height));
    uint32_t inactive_x = active_to_inactive(active_x, max_width);
    inactive_x |= hh::brev32(active_to_inactive(hh::brev32(active_x), max_width));
    inactive_y = 0 - ((inactive_y >> (threadIdx.x & 31)) & 1);

    // now incorporate the population constraint:
    uint32_t inactive_p = 0;
    if (active_p == max_pop) { inactive_p = ~active; }
    if (active_p >  max_pop) { inactive_p = 0xffffffffu; }
    uint32_t inactive = inactive_x | inactive_y | inactive_p;

    uint32_t border = get_border();
    return stator | border | (inactive & ad0 & ~exempt);
}

_DI_ uint32_t compute_next_cell(uint32_t mask, uint32_t p) {

    // printf("Thread %d has mask %d.\n", threadIdx.x, mask);

    uint32_t u_ballot = hh::ballot_32(mask != 0);

    if (u_ballot == 0) { return 0xffffffffu; }

    // compute the cells that are greater than p in reading order:
    uint32_t r = 0;
    if (threadIdx.x > ((p >> 5) & 31)) {
        r = 0xffffffffu;
    } else if (threadIdx.x == ((p >> 5) & 31)) {
        r = (0xfffffffeu) << (p & 31);
    }

    uint32_t c_mask = mask & r;
    uint32_t c_ballot = hh::ballot_32(c_mask != 0);

    if (c_ballot == 0) {
        // there are no cells after p, so wrap around:
        c_mask = mask;
        c_ballot = u_ballot;
    }

    uint32_t upper_bits = hh::ctz32(c_ballot);
    uint32_t lower_bits = hh::ctz32(hh::shuffle_32(c_mask, upper_bits));

    uint32_t q = (upper_bits << 5) | lower_bits;
    q = ((q - p - 1) & 1023) + p + 1; // increase by between 1 and 1024
    return q;
}

template<bool Horizontal, bool Vertical>
_DI_ uint32_t expand_plane(uint32_t x) {

    uint32_t y = x;
    if constexpr (Vertical) {
        y = y | kc::shift_plane< true, 1>(x) | kc::shift_plane< true, -1>(x);
    }
    if constexpr (Horizontal) {
        y = y | kc::shift_plane<false, 1>(x) | kc::shift_plane<false, -1>(x);
    }
    return y;
}

_DI_ int32_t popc128(uint4 a) {
    return hh::popc32(a.x) + hh::popc32(a.y) + hh::popc32(a.z) + hh::popc32(a.w);
}

} // namespace kc
