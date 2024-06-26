#pragma once

#include <silk/metrics.hpp>

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

#define NNUE_BYTES 3826176

int silk_main(int active_width, int active_height, int active_pop, std::string input_filename, std::string nnue_filename, int num_cadical_threads, int min_report_period, int min_stable, bool exempt_existing, std::string dataset_filename);

void enheap_then_deheap(const uint64_t* hrb, uint64_t* global_counters, uint4* heap, int hrb_size, int max_elements, uint32_t* free_nodes, int prb_size);

void launch_main_kernel(
    int blocks_to_launch,

    // device-side pointers:
    const uint4* ctx, // common context for all problems
    uint4* prb, // problem ring buffer
    uint4* srb, // solution ring buffer
    int32_t* smd, // solution metadata
    uint64_t* global_counters,
    float4* nnue,
    const uint32_t* freenodes,
    uint64_t* hrb,

    // buffer sizes:
    uint32_t prb_size,
    uint32_t srb_size,
    uint32_t hrb_size,

    // problem parameters:
    int max_width,
    int max_height,
    int max_pop,
    int min_stable,
    int rollout_gens,

    // miscellaneous:
    int min_period,
    double epsilon
);
