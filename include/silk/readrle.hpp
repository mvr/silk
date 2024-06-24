#pragma once
#include "completestill.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iterator>
#include <vector>
#include <bit>

namespace kc {

inline std::string readFileIntoString(const std::string& path) {
    std::ifstream input_file(path);
    if (!input_file.is_open()) {
        throw std::runtime_error("Could not open file");
    }
    bool reading_rle = false;    
    std::string line;
    std::ostringstream rlestream;
    while (std::getline(input_file, line)) {
        if (line.empty() || (line[0] == '#')) {
            continue;
        }
        if ((line[0] == 'x') && (reading_rle == false)) {
            reading_rle = true;
        } else if (reading_rle) {
            rlestream << line;
        }
    }
    return rlestream.str();
}

inline std::vector<uint32_t> rle2cells(std::string rle) {

    std::vector<uint32_t> cells(4096);

    uint64_t x = 0; uint64_t y = 0; uint64_t count = 0;
    uint64_t colour = 0;
    for (unsigned int i = 0; i < rle.size(); i++) {
        char c = rle[i];
        if ((c >= '0') && (c <= '9')) {
            count *= 10;
            count += (c - '0');
        } else if ((c == 'b') || (c == '.')) {
            if (count == 0) { count = 1; }
            x += count; count = 0;
        } else if (c == '$') {
            if (count == 0) { count = 1; }
            y += count; x = 0; count = 0;
        } else if ((c == 'o') || ((c >= 'A') && (c <= 'X'))) {
            if (count == 0) { count = 1; }
            if (c == 'o') {
                colour = colour * 24 + 1;
            } else {
                colour = colour * 24 + (c - 'A') + 1;
            }
            for (uint64_t j = 0; j < count; j++) {
                if ((x < 64) && (y < 64)) {
                    cells[y * 64 + x] = colour;
                }
                x += 1;
            }
            count = 0; colour = 0;
        } else if ((c >= 'p') && (c <= 'z')) {
            uint64_t m = c - 'o';
            colour = colour * 11 + (m % 11);
        } else if (c == '!') { break; }
    }

    return cells;
}

inline std::vector<uint32_t> expand_cells(const std::vector<uint32_t> &cells) {

    std::vector<uint32_t> cells2;

    for (int i = 0; i < 4096; i++) {
        if (cells[i] == 6) {
            for (int j = 0; j < 4096; j++) {
                uint32_t c = cells[j];
                if (c == 6) {
                    if (j == i) { c = 3; }
                    if (j < i) { c = 0; }
                    if (j > i) { c = 2; }
                }
                cells2.push_back(c);
            }
        }
    }

    if (cells2.size() == 0) { cells2 = cells; }

    return cells2;
}

inline void swizzle_subproblem(uint32_t* res, const uint64_t* constraints, const uint64_t* perturbation) {

    int miny = 0; int maxy = 63;
    while (perturbation[miny] == 0) { miny += 1; }
    while (perturbation[maxy] == 0) { maxy -= 1; }
    uint64_t q = 0;
    for (int y = 0; y < 64; y++) {
        q |= perturbation[y];
    }
    int minx = 0; int maxx = 63;
    while (((q >> minx) & 1) == 0) { minx += 1; }
    while (((q >> maxx) & 1) == 0) { maxx -= 1; }

    int px = (((maxx + minx) >> 1) - 16) & 63;
    int py = (((maxy + miny) >> 1) - 16) & 63;

    uint64_t per[64];

    for (int y = 0; y < 64; y++) {
        int oy = (y + py) & 63;
        per[y] = perturbation[oy];
        per[y] = (per[y] >> (px & 63)) | (per[y] << ((-px) & 63));
    }

    for (int z = 0; z < 8; z++) {

        uint64_t con[64];

        for (int y = 0; y < 64; y++) {
            int oy = (y + py) & 63;
            con[y] = constraints[oy + z * 64];
            con[y] = (con[y] >> (px & 63)) | (con[y] << ((-px) & 63));
        }

        for (int y = 0; y < 32; y++) {
            int res_offset = 128 * z + 4 * y;
            res[res_offset + 0]  = con[y]; // upper-left
            res[res_offset + 1]  = con[y] >> 32; // upper-right
            res[res_offset + 2]  = con[y + 32];  // lower-left
            res[res_offset + 3]  = con[y + 32] >> 32; // lower-right
        }
    }

    res[1024] = 1;
    res[1026] = px;
    res[1027] = py;
    res[1030] = 0;
    res[1031] = ~0;

    for (int y = 0; y < 32; y++) {
        res[1056 + y] = per[y];
    }
}

struct ProblemHolder {

    int num_subproblems;
    std::vector<uint64_t> stator;
    std::vector<uint64_t> exempt;
    std::vector<uint64_t> perturbation;
    std::vector<uint64_t> constraints;

    ProblemHolder(const std::string &filename) : num_subproblems(0), stator(64) {

        auto cells = rle2cells(readFileIntoString(filename));

        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                int c = cells[y * 64 + x];
                if ((c == 4) || (c == 5)) { stator[y] |= (1ull << x); }
            }
        }

        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                int c = cells[y * 64 + x];
                if (c == 3) {
                  // mvrnote: there must be a less dumb way to do this
                  exempt[(y+63) % 64] |= hh::rotl64(0b111ULL, (x+63) % 64);
                  exempt[y          ] |= hh::rotl64(0b111ULL, (x+63) % 64);
                  exempt[(y+1)  % 64] |= hh::rotl64(0b111ULL, (x+63) % 64);
                }
            }
        }


        cells = expand_cells(cells);

        num_subproblems = cells.size() >> 12;

        std::cerr << "Info: loaded " << num_subproblems << " subproblems." << std::endl;

        for (int z = 0; z < num_subproblems; z++) {

            std::vector<uint64_t> known_live(64);
            std::vector<uint64_t> known_dead(64);

            for (int y = 0; y < 64; y++) {
                uint64_t p = 0;
                for (int x = 0; x < 64; x++) {
                    int c = cells[z * 4096 + y * 64 + x];
                    if (c == 1) { p |= (1ull << x); }
                    if ((c == 0) || (c == 1) || (c == 4)) { known_dead[y] |= (1ull << x); }
                    if ((c == 3) || (c == 5)) { known_live[y] |= (1ull << x); }
                }
                perturbation.push_back(p);
            }

            auto local_constraints = expand_constraints(&(known_live[0]), &(known_dead[0]));
            weak_stableprop(&(local_constraints[0]));
            for (auto&& x : local_constraints) { constraints.push_back(x); }
        }
    }

    std::vector<uint64_t> complete() {
        return complete_still_life(&(constraints[0]), 4, true);
    }

    static std::vector<uint32_t> swizzle_field(std::vector<uint64_t> &field) {
        std::vector<uint32_t> res(128);
        for (int y = 0; y < 32; y++) {
            res[4 * y]     = field[y];
            res[4 * y + 1] = field[y] >> 32;
            res[4 * y + 2] = field[y + 32];
            res[4 * y + 3] = field[y + 32] >> 32;
        }
        return res;
    }

    std::vector<uint32_t> swizzle_stator() { return swizzle_field(stator); }
    std::vector<uint32_t> swizzle_exempt() { return swizzle_field(exempt); }

    std::vector<uint32_t> swizzle_problem() {
        std::vector<uint32_t> res(1120 * num_subproblems);

        for (int z = 0; z < num_subproblems; z++) {
            swizzle_subproblem(&(res[1120 * z]), &(constraints[512 * z]), &(perturbation[64 * z]));
        }

        return res;
    }

};

} // namespace kc
