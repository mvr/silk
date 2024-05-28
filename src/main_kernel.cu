#include "common.hpp"
#include <silk/mainloop.hpp>
#include <silk/readrle.hpp>

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

        // buffer sizes:
        uint32_t prb_size,
        uint32_t srb_size,

        // problem parameters:
        int max_width,
        int max_height,
        int max_pop,
        int rollout_gens,

        // miscellaneous:
        int min_period,
        uint32_t epsilon_threshold
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
    kc::load4(reading_location + 192, ad0.x, ad1.x, ad2.x, al2.x);
    kc::load4(reading_location + 224, al3.x, ad4.x, ad5.x, ad6.x);

    // load stator constraints and shift into the correct reference frame:
    uint32_t px = hh::shuffle_32(metadata_y, 2);
    uint32_t py = hh::shuffle_32(metadata_y, 3);

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
        perturbation, px, py, max_width, max_height, max_pop, rollout_gens,
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
    uint4* writing_location = prb + uint4s_per_pp * (output_idx >> 1);

    uint32_t total_info = 0; // between 0 and 32768
    total_info += kc::save4(writing_location,       ad0.y, ad1.y, ad2.y, al2.y);
    total_info += kc::save4(writing_location + 32,  al3.y, ad4.y, ad5.y, ad6.y);
    total_info += kc::save4(writing_location + 64,  ad0.z, ad1.z, ad2.z, al2.z);
    total_info += kc::save4(writing_location + 96,  al3.z, ad4.z, ad5.z, ad6.z);
    total_info += kc::save4(writing_location + 128, ad0.w, ad1.w, ad2.w, al2.w);
    total_info += kc::save4(writing_location + 160, al3.w, ad4.w, ad5.w, ad6.w);
    total_info += kc::save4(writing_location + 192, ad0.x, ad1.x, ad2.x, al2.x);
    total_info += kc::save4(writing_location + 224, al3.x, ad4.x, ad5.x, ad6.x);
    total_info = hh::warp_add(total_info);

    __syncthreads();

    // ********** PERFORM HARD BRANCHING DECISION **********

    uint32_t metadata_z = 0;

    uint32_t metadata_out = 0;

    float final_loss = kc::hard_branch(
        writing_location, perturbation, metadata_z, metadata_out,
        ad0.x, ad1.x, ad2.x, al2.x, al3.x, ad4.x, ad5.x, ad6.x, stator.x,
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

/**
 * Rather trivial kernel that produces training data from the
 * output of computecellorbackup().
 */
__global__ void makennuedata(const uint4* prb, const uint64_t* global_counters, uint32_t* dataset, uint32_t prb_size) {

    constexpr uint64_t uint4s_per_pp = 344;

    // get the location from which to read:
    uint32_t pair_idx = global_counters[COUNTER_READING_HEAD] >> 1;
    pair_idx -= (blockIdx.x + 1);
    pair_idx &= ((prb_size >> 1) - 1);

    __shared__ uint32_t metadata[32];

    // load the metadata:
    const uint32_t* metadata_location = ((const uint32_t*) (prb + pair_idx * uint4s_per_pp + 320));
    metadata[threadIdx.x] = metadata_location[threadIdx.x];
    __syncthreads();

    uint32_t signature = metadata_location[threadIdx.x + 64];
    if (threadIdx.x == 29) { signature = 0; }
    if (threadIdx.x == 30) { signature = metadata[4]; }
    if (threadIdx.x == 31) {
        float total_loss = 0.0f;
        uint32_t info_0 = metadata[5];
        if (metadata[8]) {
            uint32_t info_gain = hh::min(metadata[13] - info_0, ((uint32_t) 20));
            float sub_loss = __int_as_float(metadata[12]);
            sub_loss = hh::max(0.0f, hh::min(sub_loss, 1.0f));
            total_loss += 0.375f + 0.125f * sub_loss - 0.015625f * info_gain;
        }
        if (metadata[16]) {
            uint32_t info_gain = hh::min(metadata[21] - info_0, ((uint32_t) 20));
            float sub_loss = __int_as_float(metadata[20]);
            sub_loss = hh::max(0.0f, hh::min(sub_loss, 1.0f));
            total_loss += 0.375f + 0.125f * sub_loss - 0.015625f * info_gain;
        }
        signature = __float_as_int(total_loss);
    }

    __syncthreads();
    dataset[pair_idx * 32 + threadIdx.x] = signature;
}

struct SilkGPU {

    // device-side pointers:
    uint4* ctx;
    uint4* prb; // problem ring buffer
    uint4* srb; // solution ring buffer
    int32_t* smd; // solution metadata
    uint64_t* global_counters;
    float4* nnue;
    uint32_t* dataset;

    // host-side pointers:
    uint64_t* host_counters;

    // buffer sizes:
    uint32_t prb_size;
    uint32_t srb_size;

    // problem parameters:
    int max_width;
    int max_height;
    int max_pop;
    int rollout_gens;

    SilkGPU(uint64_t prb_capacity, uint64_t srb_capacity) {
        cudaMalloc((void**) &ctx, 512);
        cudaMalloc((void**) &prb, 2752 * prb_capacity);
        cudaMalloc((void**) &dataset, 64 * prb_capacity);
        cudaMalloc((void**) &srb, 4096 * srb_capacity);
        cudaMalloc((void**) &smd, 4 * srb_capacity);
        cudaMalloc((void**) &global_counters, 512);
        cudaMalloc((void**) &nnue, 7627264);

        cudaMallocHost((void**) &host_counters, 512);

        prb_size = prb_capacity;
        srb_size = srb_capacity;

        for (int i = 0; i < 64; i++) { host_counters[i] = 0; }
        host_counters[COUNTER_WRITING_HEAD] = 2;

        cudaMemcpy(global_counters, host_counters, 512, cudaMemcpyHostToDevice);

        cudaMemset(ctx, 0, 512);
        cudaMemset(nnue, 0, 7627264);

        max_width = 5;
        max_height = 5;
        max_pop = 10;
        rollout_gens = 6;
    }

    ~SilkGPU() {
        cudaFree(ctx);
        cudaFree(prb);
        cudaFree(srb);
        cudaFree(smd);
        cudaFree(global_counters);
        cudaFree(nnue);
        cudaFree(dataset);
        cudaFreeHost(host_counters);
    }

    void inject_problem(std::vector<uint32_t> problem, std::vector<uint32_t> stator) {
        cudaMemcpy(ctx, &(stator[0]), 512, cudaMemcpyHostToDevice);
        cudaMemcpy(prb, &(problem[0]), 5504, cudaMemcpyHostToDevice);
    }

    void run_main_kernel(int blocks_to_launch, int min_period, double epsilon) {

        // we convert the probability epsilon into an integer in [0, 2**22]
        // as that is what the kernel expects:
        uint32_t epsilon_threshold = ((uint32_t) (epsilon * 4194304.0));

        // run the kernel:
        computecellorbackup<<<blocks_to_launch, 32>>>(
            ctx, prb, srb, smd, global_counters, nnue,
            prb_size, srb_size,
            max_width, max_height, max_pop, rollout_gens,
            min_period, epsilon_threshold
        );

        // extract training data into contiguous gmem:
        makennuedata<<<blocks_to_launch / 2, 32>>>(
            prb, global_counters, dataset, prb_size
        );

        cudaMemcpy(host_counters, global_counters, 512, cudaMemcpyDeviceToHost);
    }
};

void print_solution(const uint32_t* solution) {

    uint64_t tmp[512];
    for (int z = 0; z < 8; z++) {
        for (int y = 0; y < 32; y++) {
            tmp[64 * z + y]      = solution[128 * z + 4 * y    ] | (((uint64_t) solution[128 * z + 4 * y + 1]) << 32);
            tmp[64 * z + y + 32] = solution[128 * z + 4 * y + 2] | (((uint64_t) solution[128 * z + 4 * y + 3]) << 32);
        }
    }

    auto res = kc::complete_still_life(tmp, 4, true);

    if (res.size() == 0) { return; }

    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            std::cout << (((res[y] >> x) & 1) ? '*' : '.');
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {

    kc::ProblemHolder ph("examples/tl.rle");
    auto problem = ph.swizzle_problem();
    auto stator = ph.swizzle_stator();

    SilkGPU silk(262144, 16384);

    silk.inject_problem(problem, stator);

    int problems = 2;

    for (int j = 0; j < 120; j++) {
        silk.run_main_kernel(problems, 11, 1.0);
        for (int i = 0; i < 64; i++) {
            std::cout << silk.host_counters[i] << " ";
        }
        std::cout << std::endl;
        problems = silk.host_counters[COUNTER_WRITING_HEAD] - silk.host_counters[COUNTER_READING_HEAD];
        if (problems == 0) { break; }
    }

    uint64_t solcount = silk.host_counters[COUNTER_SOLUTION_HEAD];

    return 0;

    if (solcount > 0) {
        uint32_t* host_solutions;
        int32_t* host_smd;
        cudaMallocHost((void**) &host_solutions, 4096 * solcount);
        cudaMallocHost((void**) &host_smd, 4 * solcount);

        cudaMemcpy(host_solutions, silk.srb, 4096 * solcount, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_smd, silk.smd, 4 * solcount, cudaMemcpyDeviceToHost);

        for (uint64_t i = 0; i < solcount; i++) {
            std::cout << "***** found object with return code " << host_smd[i] << " *****" << std::endl;
            print_solution(host_solutions + 1024 * i);
        }

        cudaFree(host_solutions);
        cudaFree(host_smd);
    }

    return 0;
}
