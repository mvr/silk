#pragma once
#include <cedilla/solver.hpp>
#include "hilbert.hpp"

namespace kc {

template<typename T>
inline std::vector<T> expand_constraints(const T* known_live, const T* known_dead) {

    int H = 8 * sizeof(T);
    std::vector<T> constraints(H*8);

    for (int i = 0; i < H; i++) {
        constraints[i] = known_live[i];
        constraints[i + H] = known_live[i];
        constraints[i + 2*H] = known_live[i];
        constraints[i + 3*H] = known_dead[i];
        constraints[i + 4*H] = known_dead[i];
        constraints[i + 5*H] = known_live[i];
        constraints[i + 6*H] = known_live[i];
        constraints[i + 7*H] = known_live[i];
    }

    return constraints;
}

template<typename T>
inline void weak_stableprop(T* constraints) {

    T diff = 0;

    do {

        // grid size:
        int H = 8 * sizeof(T);

        std::vector<T> known_live(H);
        std::vector<T> known_dead(H);

        // determine known cells:
        for (int i = 0; i < H; i++) {
            known_live[i] = constraints[i] & constraints[i + H] & constraints[i + 2 * H];
            known_dead[i] = constraints[i + 3 * H] & constraints[i + 4 * H];
            known_live[i] &= constraints[i + 5 * H] & constraints[i + 6 * H] & constraints[i + 7 * H];
        }

        std::vector<T> nc(8*H);

        for (int i = 0; i < 8*H; i++) { nc[i] = constraints[i]; }

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < H; x++) {
                int dead_count = 0;
                int live_count = 0;
                for (int oy = y - 1; oy <= y + 1; oy++) {
                    for (int ox = x - 1; ox <= x + 1; ox++) {
                        if ((oy == y) && (ox == x)) { continue; }
                        live_count += (known_live[oy & (H - 1)] >> (ox & (H - 1))) & 1;
                        dead_count += (known_dead[oy & (H - 1)] >> (ox & (H - 1))) & 1;
                    }
                }
                nc[y]         |=  (live_count >= 1)                       ? (((T) 1) << x) : 0;
                nc[y +     H] |= ((live_count >= 2) || (dead_count >= 8)) ? (((T) 1) << x) : 0;
                nc[y + 2 * H] |= ((live_count >= 3) || (dead_count >= 7)) ? (((T) 1) << x) : 0;
                nc[y + 3 * H] |= ((live_count >= 3) || (dead_count >= 7)) ? (((T) 1) << x) : 0;
                nc[y + 4 * H] |= ((live_count >= 4) || (dead_count >= 6)) ? (((T) 1) << x) : 0;
                nc[y + 5 * H] |= ((live_count >= 5) || (dead_count >= 5)) ? (((T) 1) << x) : 0;
                nc[y + 6 * H] |= ((live_count >= 6) || (dead_count >= 4)) ? (((T) 1) << x) : 0;
                nc[y + 7 * H] |=                       (dead_count >= 3)  ? (((T) 1) << x) : 0;

                int max_live = 0;
                if (((nc[y +     H] >> x) & 1) == 0) { max_live = 1; }
                if (((nc[y + 2 * H] >> x) & 1) == 0) { max_live = 2; }
                if (((nc[y + 3 * H] >> x) & 1) == 0) { max_live = 2; }
                if (((nc[y + 4 * H] >> x) & 1) == 0) { max_live = 3; }
                if (((nc[y + 5 * H] >> x) & 1) == 0) { max_live = 4; }
                if (((nc[y + 6 * H] >> x) & 1) == 0) { max_live = 5; }
                if (((nc[y + 7 * H] >> x) & 1) == 0) { max_live = 6; }

                int max_dead = 0;
                if (((nc[y + 7 * H] >> x) & 1) == 0) { max_dead = 2; }
                if (((nc[y + 6 * H] >> x) & 1) == 0) { max_dead = 3; }
                if (((nc[y + 5 * H] >> x) & 1) == 0) { max_dead = 4; }
                if (((nc[y + 4 * H] >> x) & 1) == 0) { max_dead = 5; }
                if (((nc[y + 3 * H] >> x) & 1) == 0) { max_dead = 6; }
                if (((nc[y + 2 * H] >> x) & 1) == 0) { max_dead = 6; }
                if (((nc[y +     H] >> x) & 1) == 0) { max_dead = 7; }
                if (((nc[y]         >> x) & 1) == 0) { max_dead = 8; }

                for (int oy = y - 1; oy <= y + 1; oy++) {
                    for (int ox = x - 1; ox <= x + 1; ox++) {
                        if ((oy == y) && (ox == x)) { continue; }
                        if ((((known_live[oy & (H - 1)] >> (ox & (H - 1))) & 1) == 0) && (live_count == max_live)) {
                            known_dead[oy & (H - 1)] |= (((T) 1) << (ox & (H - 1)));
                        }
                        if ((((known_dead[oy & (H - 1)] >> (ox & (H - 1))) & 1) == 0) && (dead_count == max_dead)) {
                            known_live[oy & (H - 1)] |= (((T) 1) << (ox & (H - 1)));
                        }
                    }
                }
            }
        }

        for (int i = 0; i < H; i++) {
            nc[i]         |= known_live[i];
            nc[i +     H] |= known_live[i];
            nc[i + 2 * H] |= known_live[i];
            nc[i + 3 * H] |= known_dead[i];
            nc[i + 4 * H] |= known_dead[i];
            nc[i + 5 * H] |= known_live[i];
            nc[i + 6 * H] |= known_live[i];
            nc[i + 7 * H] |= known_live[i];
        }

        diff = 0;

        for (int i = 0; i < 8*H; i++) { diff |= constraints[i] ^ nc[i]; constraints[i] = nc[i]; }

    } while (diff != 0);
}


template<typename T>
inline std::vector<T> complete_still_life(const T* constraints, int radius = 4, bool minimise = false) {

    // grid size:
    int H = 8 * sizeof(T);

    std::vector<std::vector<int32_t>> vars(H);

    std::vector<T> c1(H);
    std::vector<T> c2(H);

    for (int i = 0; i < H; i++) { c1[i] = constraints[i]; }
    for (int j = 0; j < radius; j++) {
        for (int i = 0; i < H; i++) {
            c2[i] = c1[i] | (c1[i] << 1) | (c1[i] >> (H-1)) | (c1[i] >> 1) | (c1[i] << (H-1));
        }
        for (int i = 0; i < H; i++) {
            c1[i] = c2[i] | c2[(i + 1) & (H - 1)] | c2[(i - 1) & (H - 1)];
        }
    }

    // determine known cells:
    for (int i = 0; i < H; i++) {
        T known_live = constraints[i] & constraints[i + H] & constraints[i + 2 * H];
        T known_dead = constraints[i + 3 * H] & constraints[i + 4 * H];
        known_live &= constraints[i + 5 * H] & constraints[i + 6 * H] & constraints[i + 7 * H];
        known_dead |= (~c1[i]);

        for (int j = 0; j < H; j++) {
            int32_t var = 0;
            if ((known_dead >> j) & 1) { var = 1; }
            if ((known_live >> j) & 1) { var = -1; }
            vars[i].push_back(var);
        }
    }

    hh::cedilla ced;

    // create variables for unknown cells, with a Hilbert curve order
    // so that numerically proximate variables correspond to spatially
    // proximate cells:
    std::vector<uint64_t> locations;
    std::vector<int32_t> unknown_lits;
    for (int p = 0; p < H * H; p++) {
        uint64_t z = kc::hilbert_d2xy(p);
        int32_t y = ((int32_t) (z >> 32));
        int32_t x = ((int32_t) (z & 0xffffffffu));

        if (vars[y][x] == 0) {
            vars[y][x] = ced.new_literal(true, minimise);
            locations.push_back(z);
            unknown_lits.push_back(vars[y][x]);
            // std::cout << "cell (" << y << ", " << x << ") unknown" << std::endl;
        }
    }

    std::vector<int32_t> unknown_lits_sorted;

    if (minimise) { unknown_lits_sorted = ced.cnf.sort_booleans(unknown_lits); }

    // add constraints:
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < H; x++) {
            std::vector<int32_t> v;
            for (int y2 = y - 1; y2 <= y + 1; y2++) {
                for (int x2 = x - 1; x2 <= x + 1; x2++) {
                    // torus:
                    v.push_back(vars[y2 & (H - 1)][x2 & (H - 1)]);
                }
            }
            ced.add_proposition(v, [&](uint64_t p) {
                uint64_t this_state = (p >> 4) & 1;
                int count = hh::popc64(p & 495);
                if ((count == 0) && (this_state == 0)) {
                    return (((constraints[y] >> x) & 1) == 0);
                } else if ((count == 1) && (this_state == 0)) {
                    return (((constraints[y + H] >> x) & 1) == 0);
                } else if ((count == 2) && (this_state == 0)) {
                    return (((constraints[y + 2 * H] >> x) & 1) == 0);
                } else if ((count == 2) && (this_state == 1)) {
                    return (((constraints[y + 3 * H] >> x) & 1) == 0);
                } else if ((count == 3) && (this_state == 1)) {
                    return (((constraints[y + 4 * H] >> x) & 1) == 0);
                } else if ((count == 4) && (this_state == 0)) {
                    return (((constraints[y + 5 * H] >> x) & 1) == 0);
                } else if ((count == 5) && (this_state == 0)) {
                    return (((constraints[y + 6 * H] >> x) & 1) == 0);
                } else if ((count == 6) && (this_state == 0)) {
                    return (((constraints[y + 7 * H] >> x) & 1) == 0);
                } else {
                    return false;
                }
            });
        }
    }

    bool has_solution = false;

    int lower_bound = 0;
    int upper_bound = locations.size() + 1;
    int middle_bound = (upper_bound + lower_bound) >> 1;

    ced.multisolve([&](const std::vector<int32_t> &solution) {

        if (solution.size() < locations.size()) {
            // this means that we made an assumption and received no solution.
            upper_bound = middle_bound;

            if (upper_bound >= lower_bound + 2) {
                middle_bound = (upper_bound + lower_bound) >> 1;
                ced.cnf.assume(-unknown_lits_sorted[middle_bound]);
            }

            return;
        }

        for (size_t k = 0; k < solution.size(); k++) {
            uint64_t z = locations[k];
            int32_t y = ((int32_t) (z >> 32));
            int32_t x = ((int32_t) (z & 0xffffffffu));
            vars[y][x] = (solution[k] > 0) ? -1 : 1;
        }
        has_solution = true;
        if (minimise) {
            int livepop = 0;
            int deadpop = 0;
            for (size_t k = 0; k < solution.size(); k++) {
                if (solution[k] > 0) {
                    livepop += 1;
                } else {
                    deadpop += 1;
                }
            }
            // std::cout << "live population " << livepop << "; dead population " << deadpop << std::endl;
            if (livepop >= 1) {
                ced.add_clause({-unknown_lits_sorted[deadpop]});
                lower_bound = deadpop;
                if (upper_bound >= lower_bound + 3) {
                    middle_bound = (upper_bound + lower_bound + 1) >> 1;
                    ced.cnf.assume(-unknown_lits_sorted[middle_bound - 1]);
                }
            } else {
                ced.add_clause({1});
            }
        }
    });

    std::vector<T> solution;

    if (has_solution) {
        for (int y = 0; y < H; y++) {
            T p = 0;
            for (int x = 0; x < H; x++) {
                if (vars[y][x] == -1) {
                    p |= ((T) 1) << x;
                }
            }
            solution.push_back(p);
        }
    }

    return solution;
}

} // namespace kc
