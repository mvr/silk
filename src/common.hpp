#pragma once

#include <cpads/core.hpp>

#define COUNTER_READING_HEAD 32
#define COUNTER_WRITING_HEAD 33
#define COUNTER_SOLUTION_HEAD 34

// We use 4480 bytes (280 uint4s) to represent a pair of problems:
//  -- bytes [0:4096]: stable information
//      -- bytes [0:1024]: upper-right quadrant
//      -- bytes [1024:2048]: lower-left quadrant
//      -- bytes [2048:3072]: lower-right quadrant
//      -- bytes [3072:4096]: upper-left quadrant
//  -- bytes [4096:4224]: metadata
//  -- bytes [4224:4352]: active perturbation
//  -- bytes [4352:4480]: signature of cell where split occurred
#define PROBLEM_PAIR_BYTES 4480
