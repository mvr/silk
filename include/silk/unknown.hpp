#pragma once
#include "stable.hpp"

namespace kc {

/**
 * Returns the sum of the 16 booleans obtained by taking the
 * values of a and b on each of the 8 neighbours. For our application
 * (computing lower and upper bounds on the change, versus the stable
 * state, of the number of live cells in the neighbourhood) this can
 * saturate at 15 without any bad consequences.
 */
template<bool SubtractOne>
_DI_ uint4 sum16(uint32_t ia, uint32_t ib) {

    uint32_t a, b;

    if constexpr (SubtractOne) {
        a = ~ia; b = ~ib;
    } else {
        a = ia; b = ib;
    }

    // 4 warp shuffles:
    uint32_t al = kc::shift_plane<true,  1>(a);
    uint32_t ar = kc::shift_plane<true, -1>(a);
    uint32_t bl = kc::shift_plane<true,  1>(b);
    uint32_t br = kc::shift_plane<true, -1>(b);

    uint32_t o2 = apply_maj3(al, ar, bl);
    uint32_t o1 = apply_xor3(al, ar, bl);

    // the sum of the upper and lower cells is in {o2, o1, br}

    uint32_t p2 = apply_maj3(o1, a, b);
    uint32_t p1 = apply_xor3(o1, a, b);

    // the sum of this column of 3 is in {o2, p2, p1, br}

    // 8 shifts:
    uint32_t o2u = kc::shift_plane<false,  1>(o2);
    uint32_t p2u = kc::shift_plane<false,  1>(p2);
    uint32_t p1u = kc::shift_plane<false,  1>(p1);
    uint32_t bru = kc::shift_plane<false,  1>(br);
    uint32_t o2d = kc::shift_plane<false, -1>(o2);
    uint32_t p2d = kc::shift_plane<false, -1>(p2);
    uint32_t p1d = kc::shift_plane<false, -1>(p1);
    uint32_t brd = kc::shift_plane<false, -1>(br);

    uint32_t q2 = apply_maj3(bru, brd, br);
    uint32_t q1 = apply_xor3(bru, brd, br);
    uint32_t r2 = apply_maj3(p1u, p1d, o1);
    uint32_t r1 = apply_xor3(p1u, p1d, o1);

    // the sum of the neighbourhood is in {o2u, p2u, o2d, p2d, o2, q2, r2, q1, r1}
    // compute lowest bit and carry:
    uint32_t s2 = q1 & r1;
    uint32_t t4 = apply_maj3(o2u, o2d, o2);
    uint32_t t2 = apply_xor3(o2u, o2d, o2);
    uint32_t u4 = apply_maj3(p2u, p2d, q2);
    uint32_t u2 = apply_xor3(p2u, p2d, q2);
    uint32_t v4 = apply_maj3(r2, s2, t2);
    uint32_t v2 = apply_xor3(r2, s2, t2);

    // the sum of the neighbourhood is in {t4, u4, v4, u2, v2, q1 ^ r1}
    // compute next bit and carry:
    uint32_t w4 = u2 & v2;
    uint32_t x8 = apply_maj3(t4, u4, v4);
    uint32_t x4 = apply_xor3(t4, u4, v4);

    // the sum of the neighbourhood is in {x8, x4, w4, u2 ^ v2, q1 ^ r1}
    // determine whether we have overflowed:
    uint32_t saturated = x8 & x4 & w4;

    // compute result in binary:
    uint4 res;
    res.x = (q1 ^ r1) | saturated;
    res.y = (u2 ^ v2) | saturated;
    res.z = (x4 ^ w4) | saturated;
    res.w = x8 | (x4 & w4);

    if constexpr (SubtractOne) {
        res.x = ~res.x;
        res.y = ~res.y;
        res.z = ~res.z;
        res.w = ~res.w;
    }

    return res;
}

#include "generated/advance.hpp"

} // namespace kc
