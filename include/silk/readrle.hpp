#pragma once
#include "completestill.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iterator>
#include <vector>

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
        // std::cout << line << std::endl;
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

struct ProblemHolder {

    std::vector<uint64_t> stator;
    std::vector<uint64_t> perturbation;
    std::vector<uint64_t> constraints;

    ProblemHolder(const std::string &filename) : stator(64), perturbation(64) {

        auto cells = rle2cells(readFileIntoString(filename));

        std::vector<uint64_t> known_live(64);
        std::vector<uint64_t> known_dead(64);

        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                int c = cells[y * 64 + x];
                if (c == 1) { perturbation[y] |= (1ull << x); }
                if ((c == 0) || (c == 1) || (c == 4)) { known_dead[y] |= (1ull << x); }
                if ((c == 3) || (c == 5)) { known_live[y] |= (1ull << x); }
                if ((c == 4) || (c == 5)) { stator[y] |= (1ull << x); }
            }
        }

        constraints = expand_constraints(&(known_live[0]), &(known_dead[0]));
        weak_stableprop(&(constraints[0]));
    }

    std::vector<uint64_t> complete() {
        return complete_still_life(&(constraints[0]), 4, true);
    }

    std::vector<uint32_t> swizzle_stator() {
        std::vector<uint32_t> res(128);
        for (int y = 0; y < 32; y++) {
            res[4 * y]     = stator[y];
            res[4 * y + 1] = stator[y] >> 32;
            res[4 * y + 2] = stator[y + 32];
            res[4 * y + 3] = stator[y + 32] >> 32;
        }
        return res;
    }

    std::vector<uint32_t> swizzle_problem() {
        std::vector<uint32_t> res(1376);

        for (int z = 0; z < 8; z++) {
            for (int y = 0; y < 32; y++) {
                int res_offset = 4 * y + (z & 3) + (z & 4) * 32;
                int con_offset = 64 * z + y;
                res[res_offset]        = constraints[con_offset] >> 32; // upper-right
                res[res_offset + 256]  = constraints[con_offset + 32];  // lower-left
                res[res_offset + 512]  = constraints[con_offset + 32] >> 32; // lower-right
                res[res_offset + 768]  = constraints[con_offset]; // upper-left
                res[res_offset + 1024] = constraints[con_offset]; // upper-left
            }
        }

        res[1280] = 1;

        for (int y = 0; y < 32; y++) {
            res[1312 + y] = perturbation[y];
        }

        return res;
    }

};

} // namespace kc
