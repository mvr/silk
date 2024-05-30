#include "common.hpp"
#include <silk/readrle.hpp>

/**
 * Rather trivial kernel that produces training data from the
 * output of computecellorbackup().
 */
__global__ void makennuedata(const uint4* prb, const uint64_t* global_counters, uint32_t* dataset, uint32_t prb_size) {

    constexpr uint64_t uint4s_per_pp = PROBLEM_PAIR_BYTES >> 4;

    // get the location from which to read:
    uint32_t pair_idx = global_counters[COUNTER_READING_HEAD] >> 1;
    pair_idx -= (blockIdx.x + 1);
    pair_idx &= ((prb_size >> 1) - 1);

    __shared__ uint32_t metadata[32];

    // load the metadata:
    const uint32_t* metadata_location = ((const uint32_t*) (prb + pair_idx * uint4s_per_pp + (uint4s_per_pp - 24)));
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
    uint32_t* freenodes;
    uint64_t* hrb;
    uint4* heap;

    // host-side pointers:
    uint64_t* host_counters;
    uint32_t* host_freenodes;

    // buffer sizes:
    uint32_t prb_size;
    uint32_t srb_size;
    uint32_t hrb_size;

    // problem parameters:
    int max_width;
    int max_height;
    int max_pop;
    int rollout_gens;

    SilkGPU(uint64_t prb_capacity, uint64_t srb_capacity) {

        uint64_t hrb_capacity = prb_capacity >> 4;

        cudaMalloc((void**) &ctx, 512);
        cudaMalloc((void**) &prb, (PROBLEM_PAIR_BYTES >> 1) * prb_capacity);
        cudaMalloc((void**) &dataset, 268435456);
        cudaMalloc((void**) &srb, 4096 * srb_capacity);
        cudaMalloc((void**) &smd, 4 * srb_capacity);
        cudaMalloc((void**) &global_counters, 512);
        cudaMalloc((void**) &nnue, 7627264);

        cudaMalloc((void**) &freenodes, 2 * prb_capacity);
        cudaMallocHost((void**) &host_freenodes, 2 * prb_capacity);

        cudaMalloc((void**) &hrb, 8 * hrb_capacity);
        cudaMalloc((void**) &heap, 8 * prb_capacity);

        cudaMallocHost((void**) &host_counters, 512);

        prb_size = prb_capacity;
        srb_size = srb_capacity;
        hrb_size = hrb_capacity;

        for (int i = 0; i < ((int) (prb_capacity >> 1)); i++) { host_freenodes[i] = i; }
        for (int i = 0; i < 64; i++) { host_counters[i] = 0; }

        cudaMemcpy(freenodes, host_freenodes, 2 * prb_capacity, cudaMemcpyHostToDevice);

        cudaMemset(ctx, 0, 512);
        cudaMemset(nnue, 0, 7627264);

        max_width = 8;
        max_height = 8;
        max_pop = 16;
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

        int num_problems = (4 * problem.size()) / PROBLEM_PAIR_BYTES;

        host_counters[COUNTER_WRITING_HEAD] = 2 * num_problems;
        host_counters[COUNTER_MIDDLE_HEAD] = 2 * num_problems;

        cudaMemcpy(global_counters, host_counters, 512, cudaMemcpyHostToDevice);
        cudaMemcpy(ctx, &(stator[0]), 512, cudaMemcpyHostToDevice);
        cudaMemcpy(prb, &(problem[0]), PROBLEM_PAIR_BYTES * num_problems, cudaMemcpyHostToDevice);
    }

    void run_main_kernel(int blocks_to_launch, int min_period, double epsilon, int max_batch_size) {

        // we convert the probability epsilon into an integer in [0, 2**22]
        // as that is what the kernel expects:
        uint32_t epsilon_threshold = ((uint32_t) (epsilon * 4194304.0));

        // run the kernel:
        launch_main_kernel(blocks_to_launch,
            ctx, prb, srb, smd, global_counters, nnue, freenodes, hrb,
            prb_size, srb_size, hrb_size,
            max_width, max_height, max_pop, rollout_gens,
            min_period, epsilon_threshold
        );

        /*
        // extract training data into contiguous gmem:
        makennuedata<<<blocks_to_launch / 2, 32>>>(
            prb, global_counters, dataset, prb_size
        );
        */

        enheap_then_deheap(hrb, global_counters, heap, hrb_size, max_batch_size >> 12, freenodes, prb_size);

        cudaMemcpy(host_counters, global_counters, 512, cudaMemcpyDeviceToHost);
    }
};

void print_solution(const uint32_t* solution, const uint64_t* perturbation) {

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
            std::cout << (((perturbation[y] >> x) & 1) ? 'o' : (((res[y] >> x) & 1) ? '*' : '.'));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {

    kc::ProblemHolder ph("examples/2c3.rle");
    auto problem = ph.swizzle_problem();
    auto stator = ph.swizzle_stator();

    SilkGPU silk(524288, 16384);

    silk.inject_problem(problem, stator);

    while (true) {
        int problems = silk.host_counters[COUNTER_MIDDLE_HEAD] - silk.host_counters[COUNTER_READING_HEAD];
        int total_problems = silk.host_counters[COUNTER_WRITING_HEAD] - silk.host_counters[COUNTER_READING_HEAD];

        int lower_batch_size = 4096;
        int upper_batch_size = (silk.prb_size >> 4) - 4096;
        int medium_batch_size = ((3 * silk.prb_size) >> 5) - (total_problems >> 3);

        int batch_size = hh::max(lower_batch_size, hh::min(medium_batch_size, upper_batch_size));
        batch_size &= 0x7ffff000;

        std::cout << "Open problems: \033[31;1m" << total_problems << "\033[0m; batch size: \033[32;1m" << batch_size << "\033[0m" << std::endl;
        silk.run_main_kernel(problems, 9999, 1.0, batch_size);
        for (int i = 0; i < 64; i++) {
            std::cout << silk.host_counters[i] << " ";
        }
        std::cout << std::endl;
        if (problems == 0) { break; }
    }

    uint64_t solcount = silk.host_counters[COUNTER_SOLUTION_HEAD];

    /*
    uint64_t ppcount = silk.host_counters[COUNTER_READING_HEAD] >> 1;

    {
        uint32_t* host_dataset;
        cudaMallocHost((void**) &host_dataset, 128 * ppcount);
        cudaMemcpy(host_dataset, silk.dataset, 128 * ppcount, cudaMemcpyDeviceToHost);

        std::vector<uint64_t> nncounts(512);
        for (uint64_t i = 1; i < ppcount; i++) {
            for (uint64_t j = 0; j < 29; j++) {
                nncounts[host_dataset[i * 32 + j]] += 1;
            }
        }

        for (int i = 0; i < 512; i++) {
            if (nncounts[i] != 0) {
                std::cout << i << ": " << nncounts[i] << std::endl;
            }
        }

        cudaFree(host_dataset);
    }
    */

    // return 0;

    if (solcount > 0) {
        uint32_t* host_solutions;
        int32_t* host_smd;
        cudaMallocHost((void**) &host_solutions, 4096 * solcount);
        cudaMallocHost((void**) &host_smd, 4 * solcount);

        cudaMemcpy(host_solutions, silk.srb, 4096 * solcount, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_smd, silk.smd, 4 * solcount, cudaMemcpyDeviceToHost);

        for (uint64_t i = 0; i < solcount; i++) {
            std::cout << "***** found object with return code " << host_smd[i] << " *****" << std::endl;
            print_solution(host_solutions + 1024 * i, &(ph.perturbation[0]));
        }

        cudaFree(host_solutions);
        cudaFree(host_smd);
    }

    return 0;
}
