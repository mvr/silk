#pragma once
#include "torus.hpp"

namespace kc {

_DI_ uint32_t apply_maj3(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t res; // = (x & y) | (y & z) | (z & x);
    asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(res) : "r"(x), "r"(y), "r"(z));
    return res;
}

_DI_ uint32_t apply_xor3(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t res; // = x ^ y ^ z;
    asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(res) : "r"(x), "r"(y), "r"(z));
    return res;
}

#include "generated/domino.hpp"

_DI_ bool stableprop(uint32_t &ad0, uint32_t &ad1, uint32_t &ad2, uint32_t &al2, uint32_t &al3, uint32_t &ad4, uint32_t &ad5, uint32_t &ad6) {

    {
        uint32_t contradiction = ad0 & ad1 & ad2 & al2 & al3 & ad4 & ad5 & ad6;
        contradiction = hh::ballot_32(contradiction != 0);
        if (contradiction) { return true; }
    }

    int no_progress = 0;

    while (no_progress < 2) {

        if (apply_domino_rule<false>(ad0, ad1, ad2, al2, al3, ad4, ad5, ad6)) {
            no_progress = 0;
            uint32_t contradiction = ad0 & ad1 & ad2 & al2 & al3 & ad4 & ad5 & ad6;
            contradiction = hh::ballot_32(contradiction != 0);
            if (contradiction) { return true; }
        } else {
            no_progress += 1;
        }

        if (no_progress >= 2) { break; }

        if (apply_domino_rule<true>(ad0, ad1, ad2, al2, al3, ad4, ad5, ad6)) {
            no_progress = 0;
            uint32_t contradiction = ad0 & ad1 & ad2 & al2 & al3 & ad4 & ad5 & ad6;
            contradiction = hh::ballot_32(contradiction != 0);
            if (contradiction) { return true; }
        } else {
            no_progress += 1;
        }
    }

    return false;
}


/**
 * Returns the mod-16 sum of the 16 booleans obtained by taking the
 * values of a and b on each of the 8 neighbours. For our application
 * (computing lower and upper bounds on the change, versus the stable
 * state, of the number of live cells in the neighbourhood) the change
 * is bounded in [-6, +8] so we only need to know the bound mod 16.
 */
_DI_ uint4 sum16(uint32_t a, uint32_t b) {

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

    uint4 res;

    // the sum of the neighbourhood is in {o2u, p2u, o2d, p2d, o2, q2, r2, q1, r1}
    // compute lowest bit and carry:
    res.x = q1 ^ r1;
    uint32_t s2 = q1 & r1;
    uint32_t t4 = apply_maj3(o2u, o2d, o2);
    uint32_t t2 = apply_xor3(o2u, o2d, o2);
    uint32_t u4 = apply_maj3(p2u, p2d, q2);
    uint32_t u2 = apply_xor3(p2u, p2d, q2);
    uint32_t v4 = apply_maj3(r2, s2, t2);
    uint32_t v2 = apply_xor3(r2, s2, t2);

    // the sum of the neighbourhood is in {t4, u4, v4, u2, v2, res.x}
    // compute next bit and carry:
    res.y = u2 ^ v2;
    uint32_t w4 = u2 & v2;
    uint32_t x8 = apply_maj3(t4, u4, v4);
    uint32_t x4 = apply_xor3(t4, u4, v4);

    // the sum of the neighbourhood is in {x8, x4, w4, res.y, res.x}
    // compute upper two bits:
    res.z = x4 ^ w4;
    res.w = x8 ^ (x4 & w4);

    return res;
}

}
