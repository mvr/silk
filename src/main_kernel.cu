#include <silk/mainloop.hpp>

#define COUNTER_READING_HEAD 32
#define COUNTER_WRITING_HEAD 33
#define COUNTER_SOLUTION_HEAD 34

__global__ void __launch_bounds__(32, 16) computecellorbackup(
        const uint4* ctx, // common context for all problems
        uint4* prb, // problem ring buffer
        uint32_t prb_size,
        uint4* srb, // solution ring buffer
        int32_t* smd, // solution metadata
        uint32_t srb_size,
        uint64_t* global_counters,
        int max_width, int max_height, int max_pop, int gens, int min_period,
        uint32_t epsilon_threshold, const float4* nnue
    ) {

    // We use 5504 bytes (344 uint4s) to represent a pair of problems:
    //  -- bytes [0:5120]: stable information
    //      -- bytes [0:1024]: upper-right quadrant
    //      -- bytes [1024:2048]: lower-left quadrant
    //      -- bytes [2048:3072]: lower-right quadrant
    //      -- bytes [3072:4096]: upper-left quadrant in first problem
    //      -- bytes [4096:5120]: upper-left quadrant in second problem
    //  -- bytes [5120:5248]: metadata
    //  -- bytes [5248:5376]: active perturbation
    //  -- bytes [5376:5504]: signature of cell where split occurred
    constexpr uint64_t uint4s_per_pp = 344;

    // determine which problem to load:
    uint32_t logical_block_idx = 0;
    if (threadIdx.x == 0) {
        logical_block_idx = hh::atomic_add(global_counters + COUNTER_READING_HEAD, 1) & (prb_size - 1);
    }
    logical_block_idx = hh::shuffle_32(logical_block_idx, 0);
    uint4* reading_location = prb + uint4s_per_pp * (logical_block_idx >> 1);
    uint32_t block_parity = logical_block_idx & 1;

    // ********** LOAD PROBLEM **********

    // load problem metadata:
    uint32_t* problem_metadata_location = ((uint32_t*) (reading_location + 320));
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
    uint4 ad0, ad1, ad2, al2, al3, ad4, ad5, ad6;
    kc::load4(reading_location,       ad0.y, ad1.y, ad2.y, al2.y);
    kc::load4(reading_location + 32,  al3.y, ad4.y, ad5.y, ad6.y);
    kc::load4(reading_location + 64,  ad0.z, ad1.z, ad2.z, al2.z);
    kc::load4(reading_location + 96,  al3.z, ad4.z, ad5.z, ad6.z);
    kc::load4(reading_location + 128, ad0.w, ad1.w, ad2.w, al2.w);
    kc::load4(reading_location + 160, al3.w, ad4.w, ad5.w, ad6.w);
    kc::load4(reading_location + 192 + block_parity * 64, ad0.x, ad1.x, ad2.x, al2.x);
    kc::load4(reading_location + 224 + block_parity * 64, al3.x, ad4.x, ad5.x, ad6.x);

    // load stator constraints and shift into the correct reference frame:
    uint32_t px = hh::shuffle_32(metadata_y, 2);
    uint32_t py = hh::shuffle_32(metadata_y, 3);
    uint4 stator = ctx[threadIdx.x];
    kc::shift_torus_inplace(stator, -px, -py);

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

    int return_code = kc::mainloop(
        ad0, ad1, ad2, al2, al3, ad4, ad5, ad6, stator,
        perturbation, px, py, max_width, max_height, max_pop, gens,
        smem, metrics
    );

    if (return_code == -1) { kc::bump_counter<true>(metrics, METRIC_DEADEND); }
    if (return_code ==  0) { kc::bump_counter<true>(metrics, METRIC_FIZZLE); }
    if (return_code ==  1) { kc::bump_counter<true>(metrics, METRIC_RESTAB); }
    if (return_code >=  2) { kc::bump_counter<true>(metrics, METRIC_OSCILLATOR); }

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
            // we have found a solution:

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
        output_idx = hh::atomic_add(global_counters + COUNTER_WRITING_HEAD, 1) & ((prb_size >> 1) - 1);
    }
    output_idx = hh::shuffle_32(output_idx, 0);
    uint4* writing_location = prb + uint4s_per_pp * output_idx;

    uint32_t total_info = 0; // between 0 and 32768
    total_info += kc::save4(writing_location,       ad0.y, ad1.y, ad2.y, al2.y);
    total_info += kc::save4(writing_location + 32,  al3.y, ad4.y, ad5.y, ad6.y);
    total_info += kc::save4(writing_location + 64,  ad0.z, ad1.z, ad2.z, al2.z);
    total_info += kc::save4(writing_location + 96,  al3.z, ad4.z, ad5.z, ad6.z);
    total_info += kc::save4(writing_location + 128, ad0.w, ad1.w, ad2.w, al2.w);
    total_info += kc::save4(writing_location + 160, al3.w, ad4.w, ad5.w, ad6.w);
    total_info += hh::popc32(ad0.x);
    total_info += hh::popc32(ad1.x);
    total_info += hh::popc32(ad2.x);
    total_info += hh::popc32(al2.x);
    total_info += hh::popc32(al3.x);
    total_info += hh::popc32(ad4.x);
    total_info += hh::popc32(ad5.x);
    total_info += hh::popc32(ad6.x);
    total_info = hh::warp_add(total_info);

    __syncthreads();

    // ********** PERFORM HARD BRANCHING DECISION **********

    uint32_t metadata_z = 0;

    float final_loss = kc::hard_branch(
        writing_location, perturbation, metadata_z,
        ad0.x, ad1.x, ad2.x, al2.x, al3.x, ad4.x, ad5.x, ad6.x, stator.x,
        max_width, max_height, max_pop, smem, epsilon_threshold,
        [&](uint32_t signature) __attribute__((always_inline)) {
            float loss = kc::evaluate_nnue(signature, nnue);
            return loss;
        }, metrics
    );

    // ********** WRITE OUTPUT PROBLEM METADATA **********

    uint32_t final_loss_bits = __float_as_int(final_loss);

    uint32_t metadata_out = 0;
    if (threadIdx.x < 2) { metadata_out = 1; }
    if (threadIdx.x == 2) { metadata_out = px; }
    if (threadIdx.x == 3) { metadata_out = py; }
    if (threadIdx.x == 4) { metadata_out = final_loss_bits; }
    if (threadIdx.x == 5) { metadata_out = total_info; }

    if (threadIdx.x < 8) {
        // copy the 32-byte metadata sector into the parent problem:
        problem_metadata_location[threadIdx.x + 8 + 8 * block_parity] = metadata_out;
    }

    uint32_t* solution_metadata_location = ((uint32_t*) (writing_location + 320));
    solution_metadata_location[threadIdx.x] = metadata_out;
    solution_metadata_location[threadIdx.x + 32] = perturbation;
    solution_metadata_location[threadIdx.x + 64] = metadata_z;

    // flush metrics:
    __syncthreads();
    hh::atomic_add(global_counters + threadIdx.x, metrics[threadIdx.x]);
}

void run_main_kernel(int blocks_to_launch, const uint4* ctx, uint4* prb, uint32_t prb_size, uint4* srb, int32_t* smd, uint32_t srb_size, uint64_t* global_counters, 
                        int max_width, int max_height, int max_pop, int gens, int min_period, double epsilon, const float4* nnue) {

    // we convert the probability epsilon into an integer in [0, 2**22]
    // as that is what the kernel expects:
    uint32_t epsilon_threshold = ((uint32_t) (epsilon * 4194304.0));

    // run the kernel:
    computecellorbackup<<<blocks_to_launch, 32>>>(ctx, prb, prb_size, srb, smd, srb_size, global_counters, max_width, max_height, max_pop, gens, min_period, epsilon_threshold, nnue);

}

