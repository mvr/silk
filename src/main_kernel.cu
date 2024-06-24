#include "common.hpp"
#include <silk/mainloop.hpp>

/**
 * Main kernel that does the majority of the work.
 */
__global__ void __launch_bounds__(32, 16) computecellorbackup(

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
        uint32_t epsilon_threshold
    ) {

    constexpr uint64_t uint4s_per_pp = PROBLEM_PAIR_BYTES >> 4;

    // determine which problem to load:
    uint64_t logical_block_idx = global_counters[COUNTER_READING_HEAD] + blockIdx.x;
    if (logical_block_idx >= global_counters[COUNTER_MIDDLE_HEAD]) { return; }
    logical_block_idx &= (prb_size - 1);
    uint32_t block_parity = logical_block_idx & 1;
    uint4* reading_location = prb + uint4s_per_pp * freenodes[logical_block_idx >> 1];

    // ********** LOAD PROBLEM **********

    // load problem metadata:
    uint32_t* problem_metadata_location = ((uint32_t*) (reading_location + (uint4s_per_pp - 24)));
    uint32_t metadata_y = problem_metadata_location[threadIdx.x];

    if (hh::shuffle_32(metadata_y, block_parity) == 0) {
        // we store problems in pairs for memory compression; however,
        // initially we only inject one problem so we want the ability
        // to have elements of the pair early-exit:
        return;
    }

    // load active perturbation:
    uint32_t perturbation = problem_metadata_location[threadIdx.x + 32];

    // load stable information:
    uint4 ad0 = reading_location[threadIdx.x];
    uint4 ad1 = reading_location[threadIdx.x + 32];
    uint4 ad2 = reading_location[threadIdx.x + 64];
    uint4 al2 = reading_location[threadIdx.x + 96];
    uint4 al3 = reading_location[threadIdx.x + 128];
    uint4 ad4 = reading_location[threadIdx.x + 160];
    uint4 ad5 = reading_location[threadIdx.x + 192];
    uint4 ad6 = reading_location[threadIdx.x + 224];

    // load stator constraints and shift into the correct reference frame:
    uint32_t px = hh::shuffle_32(metadata_y, 2);
    uint32_t py = hh::shuffle_32(metadata_y, 3);

    uint32_t overall_generation = hh::shuffle_32(metadata_y, 6);
    uint32_t restored_time      = hh::shuffle_32(metadata_y, 7);
    
    {
        uint32_t best_p = hh::shuffle_32(metadata_y, 30);
        uint32_t contrib = hh::shuffle_32(metadata_y, 31);
        if (block_parity & 1) { contrib ^= 255; }
        uint32_t this_cell = (threadIdx.x == (best_p >> 5)) ? (1u << (best_p & 31)) : 0u;
        if (contrib &   1) { ad0.x |= this_cell; }
        if (contrib &   2) { ad1.x |= this_cell; }
        if (contrib &   4) { ad2.x |= this_cell; }
        if (contrib &   8) { al2.x |= this_cell; }
        if (contrib &  16) { al3.x |= this_cell; }
        if (contrib &  32) { ad4.x |= this_cell; }
        if (contrib &  64) { ad5.x |= this_cell; }
        if (contrib & 128) { ad6.x |= this_cell; }
    }

    uint4 stator = ctx[threadIdx.x];
    kc::shift_torus_inplace(stator, -px, -py);
    uint4 exempt = ctx[threadIdx.x + 32];
    kc::shift_torus_inplace(exempt, -px, -py);
    
    // ********** INITIALISE SHARED MEMORY **********

    __shared__ uint32_t smem[256];
    __shared__ uint32_t metrics[32];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        smem[32*i + threadIdx.x] = 0;
    }
    metrics[threadIdx.x] = 0;
    __syncthreads();

    // ********** PERFORM PROPAGATION AND ADVANCING **********

    kc::bump_counter<true>(metrics, METRIC_KERNEL);

    int return_code = -3;
    int max_rounds = 2;

    // main loop:
    while (return_code == -3) {
        // apply soft-branching and propagation:
        bool contradiction = kc::branched_rollout<true>(
            smem, ad0.x, ad1.x, ad2.x, al2.x, al3.x, ad4.x, ad5.x, ad6.x, perturbation, stator.x, exempt.x,
            max_width, max_height, max_pop, rollout_gens, metrics, max_rounds
        );
        if (contradiction) { return_code = -1; break; }

        // advance and perform cycle detection:
        return_code = kc::floyd_cycle<true>(
            ad0, ad1, ad2, al2, al3, ad4, ad5, ad6, stator, exempt, perturbation, px, py,
            overall_generation, restored_time, max_width, max_height, max_pop, min_stable, metrics
        );

        if (return_code == -3) { max_rounds = 1; }
    }

    if (return_code == -1) { kc::bump_counter<true>(metrics, METRIC_DEADEND); }
    if (return_code ==  0) { kc::bump_counter<true>(metrics, METRIC_FIZZLE); }
    if (return_code ==  1) { kc::bump_counter<true>(metrics, METRIC_RESTAB); }
    if (return_code ==  1000000000) { kc::bump_counter<true>(metrics, METRIC_CATALYSIS); }
    else if (return_code >=  2) { kc::bump_counter<true>(metrics, METRIC_OSCILLATOR); }

    // ********** HANDLE POTENTIAL SOLUTIONS **********

    if ((return_code >= 1) && (return_code < min_period)) {
        // we have found a restabilisation or oscillator of period
        // lower than the threshold, so this does not count as a
        // solution; modify the return code accordingly:
        return_code = -1;
    }

    __syncthreads();

    if (return_code >= -1) {
        // we have found a solution or a contradiction

        // flush metrics:
        hh::atomic_add(global_counters + threadIdx.x, metrics[threadIdx.x]);

        if (return_code >= 0) {
            // we have found a solution; write it out:
            uint32_t solution_idx = 0;
            if (threadIdx.x == 0) {
                solution_idx = hh::atomic_add(global_counters + COUNTER_SOLUTION_HEAD, 1) & (srb_size - 1);
                smd[solution_idx] = return_code;
            }
            solution_idx = hh::shuffle_32(solution_idx, 0);
            uint4* solution_location = srb + 256 * ((uint64_t) solution_idx);
            kc::shift_torus_inplace(ad0, px, py);
            solution_location[threadIdx.x] = ad0;
            kc::shift_torus_inplace(ad1, px, py);
            solution_location[threadIdx.x + 32] = ad1;
            kc::shift_torus_inplace(ad2, px, py);
            solution_location[threadIdx.x + 64] = ad2;
            kc::shift_torus_inplace(al2, px, py);
            solution_location[threadIdx.x + 96] = al2;
            kc::shift_torus_inplace(al3, px, py);
            solution_location[threadIdx.x + 128] = al3;
            kc::shift_torus_inplace(ad4, px, py);
            solution_location[threadIdx.x + 160] = ad4;
            kc::shift_torus_inplace(ad5, px, py);
            solution_location[threadIdx.x + 192] = ad5;
            kc::shift_torus_inplace(ad6, px, py);
            solution_location[threadIdx.x + 224] = ad6;
        }
        return;
    }

    // ********** WRITE BEGINNING OF OUTPUT PROBLEM PAIR **********

    uint32_t output_idx = 0;
    if (threadIdx.x == 0) {
        output_idx = hh::atomic_add(global_counters + COUNTER_WRITING_HEAD, 2) & (prb_size - 1);
    }
    output_idx = hh::shuffle_32(output_idx, 0);
    output_idx = freenodes[output_idx >> 1];
    uint4* writing_location = prb + uint4s_per_pp * output_idx;

    uint32_t total_info = 0; // between 0 and 32768
    writing_location[threadIdx.x]       = ad0; total_info += kc::popc128(ad0);
    writing_location[threadIdx.x + 32]  = ad1; total_info += kc::popc128(ad1);
    writing_location[threadIdx.x + 64]  = ad2; total_info += kc::popc128(ad2);
    writing_location[threadIdx.x + 96]  = al2; total_info += kc::popc128(al2);
    writing_location[threadIdx.x + 128] = al3; total_info += kc::popc128(al3);
    writing_location[threadIdx.x + 160] = ad4; total_info += kc::popc128(ad4);
    writing_location[threadIdx.x + 192] = ad5; total_info += kc::popc128(ad5);
    writing_location[threadIdx.x + 224] = ad6; total_info += kc::popc128(ad6);
    total_info = hh::warp_add(total_info);

    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t hrb_loc = hh::atomic_add(global_counters + COUNTER_HRB_WRITING_HEAD, 1) & (hrb_size - 1);
        uint64_t entry = (((uint64_t) total_info) << 32) | output_idx;
        hrb[hrb_loc] = entry;
    }

    // ********** PERFORM HARD BRANCHING DECISION **********

    uint32_t metadata_z = 0;

    uint32_t metadata_out = 0;

    float final_loss = kc::hard_branch(
        writing_location, perturbation, metadata_z, metadata_out,
        ad0.x, ad1.x, ad2.x, al2.x, al3.x, ad4.x, ad5.x, ad6.x, stator.x, exempt.x,
        max_width, max_height, max_pop, smem, epsilon_threshold,
        [&](uint32_t signature) __attribute__((always_inline)) {
            float loss = kc::evaluate_nnue(signature, nnue);
            return loss;
        }, metrics
    );

    // ********** WRITE OUTPUT PROBLEM METADATA **********

    uint32_t final_loss_bits = __float_as_int(final_loss);

    if (threadIdx.x < 2) { metadata_out = 1; }
    if (threadIdx.x == 2) { metadata_out = px; }
    if (threadIdx.x == 3) { metadata_out = py; }
    if (threadIdx.x == 4) { metadata_out = final_loss_bits; }
    if (threadIdx.x == 5) { metadata_out = total_info; }
    if (threadIdx.x == 6) { metadata_out = overall_generation; }
    if (threadIdx.x == 7) { metadata_out = restored_time; }

    __syncthreads();

    if (threadIdx.x < 8) {
        // copy the 32-byte metadata sector into the parent problem:
        problem_metadata_location[threadIdx.x + 8 + 8 * block_parity] = metadata_out;
    }

    uint32_t* solution_metadata_location = ((uint32_t*) (writing_location + (uint4s_per_pp - 24)));
    solution_metadata_location[threadIdx.x] = metadata_out;
    solution_metadata_location[threadIdx.x + 32] = perturbation;
    solution_metadata_location[threadIdx.x + 64] = metadata_z;

    // flush metrics:
    __syncthreads();
    hh::atomic_add(global_counters + threadIdx.x, metrics[threadIdx.x]);
}

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
) {

    // we convert the probability epsilon into an integer in [0, 2**22]
    // as that is what the kernel expects:
    uint32_t epsilon_threshold = ((uint32_t) (epsilon * 4194304.0));

    // run the kernel:
    computecellorbackup<<<blocks_to_launch, 32>>>(
        ctx, prb, srb, smd, global_counters, nnue, freenodes, hrb,
        prb_size, srb_size, hrb_size,
        max_width, max_height, max_pop, min_stable, rollout_gens,
        min_period, epsilon_threshold
    );
}
