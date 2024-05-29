#pragma once

#include <cpads/core.hpp>

#define COUNTER_READING_HEAD 32
#define COUNTER_MIDDLE_HEAD  33
#define COUNTER_WRITING_HEAD 34

#define COUNTER_HRB_READING_HEAD 35
#define COUNTER_HRB_WRITING_HEAD 36
#define COUNTER_HEAP_ELEMENTS 37

#define COUNTER_SOLUTION_HEAD 38

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

void enheap_then_deheap(const uint64_t* hrb, uint64_t* global_counters, uint4* heap, int hrb_size, int max_elements, uint32_t* free_nodes, int prb_size);
