#pragma once
#include "torus.hpp"

namespace kc {

_DI_ uint32_t apply_maj3(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t res;
    asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(res) : "r"(x), "r"(y), "r"(z));
    return res;
}

_DI_ uint32_t apply_xor3(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t res;
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

}
