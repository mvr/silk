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
inline std::vector<T> complete_still_life(const T* constraints) {

    // grid size:
    int H = 8 * sizeof(T);

    std::vector<std::vector<int32_t>> vars(H);

    // determine known cells:
    for (int i = 0; i < H; i++) {
        T known_live = constraints[i] & constraints[i + H] & constraints[i + 2 * H];
        T known_dead = constraints[i + 3 * H] & constraints[i + 4 * H];
        known_live &= constraints[i + 5 * H] & constraints[i + 6 * H] & constraints[i + 7 * H];

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
    for (int p = 0; p < H * H; p++) {
        uint64_t z = kc::hilbert_d2xy(p);
        int32_t y = ((int32_t) (z >> 32));
        int32_t x = ((int32_t) (z & 0xffffffffu));

        if (vars[y][x] == 0) {
            vars[y][x] = ced.new_literal(true, false);
            locations.push_back(z);
            // std::cout << "cell (" << y << ", " << x << ") unknown" << std::endl;
        }
    }

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

    ced.multisolve([&](const std::vector<int32_t> &solution) {
        for (size_t k = 0; k < solution.size(); k++) {
            uint64_t z = locations[k];
            int32_t y = ((int32_t) (z >> 32));
            int32_t x = ((int32_t) (z & 0xffffffffu));
            vars[y][x] = (solution[k] > 0) ? -1 : 1;
        }
        has_solution = true;
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
