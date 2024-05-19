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

}
