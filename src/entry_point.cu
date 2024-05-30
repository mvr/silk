#include "common.hpp"
#include <silk/readrle.hpp>
#include <stdio.h>

/**
 * Rather trivial kernel that produces training data from the
 * output of computecellorbackup().
 */
__global__ void makennuedata(const uint4* prb, const uint32_t* freenodes, const uint64_t* global_counters, uint8_t* dataset, uint32_t prb_size, uint32_t drb_size) {

    constexpr uint64_t uint4s_per_pp = PROBLEM_PAIR_BYTES >> 4;

    // get the location from which to read:
    uint32_t pair_idx = (global_counters[COUNTER_READING_HEAD] >> 1) + blockIdx.x;
    uint32_t node_loc = freenodes[pair_idx & ((prb_size >> 1) - 1)];
    pair_idx &= (drb_size - 1);

    __shared__ uint32_t metadata[32];

    // load the metadata:
    const uint32_t* metadata_location = ((const uint32_t*) (prb + node_loc * uint4s_per_pp + (uint4s_per_pp - 24)));
    metadata[threadIdx.x] = metadata_location[threadIdx.x];
    __syncthreads();

    uint32_t signature = metadata_location[threadIdx.x + 64];
    {
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
        int32_t loss_bits = ((int32_t) (total_loss * 16777216.0f));
        loss_bits = hh::max(((int32_t) 0), hh::min(loss_bits, ((int32_t) 0xffffff)));
        if (threadIdx.x >= 29) {
            signature = (loss_bits >> ((threadIdx.x - 29) * 8)) & 255;
        }
    }

    __syncthreads();
    dataset[pair_idx * 32 + threadIdx.x] = ((uint8_t) signature);
}

struct SilkGPU {

    // device-side pointers:
    uint4* ctx;
    uint4* prb; // problem ring buffer
    uint4* srb; // solution ring buffer
    int32_t* smd; // solution metadata
    uint64_t* global_counters;
    float4* nnue;
    uint8_t* dataset;
    uint32_t* freenodes;
    uint64_t* hrb;
    uint4* heap;

    // host-side pointers:
    uint64_t* host_counters;
    uint32_t* host_freenodes;
    uint8_t* host_dataset;

    // buffer sizes:
    uint32_t prb_size;
    uint32_t srb_size;
    uint32_t hrb_size;
    uint32_t drb_size;

    // problem parameters:
    int max_width;
    int max_height;
    int max_pop;
    int rollout_gens;

    uint64_t drb_hwm;

    SilkGPU(uint64_t prb_capacity, uint64_t srb_capacity, uint64_t drb_capacity) {

        uint64_t hrb_capacity = prb_capacity >> 4;

        cudaMalloc((void**) &ctx, 512);
        cudaMalloc((void**) &prb, (PROBLEM_PAIR_BYTES >> 1) * prb_capacity);
        cudaMalloc((void**) &dataset, 32 * drb_capacity);
        cudaMalloc((void**) &srb, 4096 * srb_capacity);
        cudaMalloc((void**) &smd, 4 * srb_capacity);
        cudaMalloc((void**) &global_counters, 512);
        cudaMalloc((void**) &nnue, NNUE_BYTES);

        cudaMalloc((void**) &freenodes, 2 * prb_capacity);
        cudaMallocHost((void**) &host_freenodes, 2 * prb_capacity);
        cudaMallocHost((void**) &host_dataset, 32 * drb_capacity);

        cudaMalloc((void**) &hrb, 8 * hrb_capacity);
        cudaMalloc((void**) &heap, 8 * prb_capacity);

        cudaMallocHost((void**) &host_counters, 512);

        prb_size = prb_capacity;
        srb_size = srb_capacity;
        hrb_size = hrb_capacity;
        drb_size = drb_capacity;

        for (int i = 0; i < ((int) (prb_capacity >> 1)); i++) { host_freenodes[i] = i; }
        for (int i = 0; i < 64; i++) { host_counters[i] = 0; }

        cudaMemcpy(freenodes, host_freenodes, 2 * prb_capacity, cudaMemcpyHostToDevice);

        cudaMemset(ctx, 0, 512);
        cudaMemset(nnue, 0, NNUE_BYTES);

        max_width = 8; // 7;
        max_height = 8; // 7;
        max_pop = 12; // 14;
        rollout_gens = 6;
        drb_hwm = 0;
    }

    ~SilkGPU() {
        cudaFree(ctx);
        cudaFree(prb);
        cudaFree(srb);
        cudaFree(smd);
        cudaFree(global_counters);
        cudaFree(nnue);
        cudaFree(dataset);
        cudaFree(freenodes);
        cudaFree(heap);
        cudaFree(hrb);
        cudaFreeHost(host_counters);
        cudaFreeHost(host_dataset);
        cudaFreeHost(host_freenodes);
    }

    void inject_problem(std::vector<uint32_t> problem, std::vector<uint32_t> stator) {

        int num_problems = (4 * problem.size()) / PROBLEM_PAIR_BYTES;

        host_counters[COUNTER_WRITING_HEAD] = 2 * num_problems;
        host_counters[COUNTER_MIDDLE_HEAD] = 2 * num_problems;
        drb_hwm = host_counters[COUNTER_MIDDLE_HEAD];

        cudaMemcpy(global_counters, host_counters, 512, cudaMemcpyHostToDevice);
        cudaMemcpy(ctx, &(stator[0]), 512, cudaMemcpyHostToDevice);
        cudaMemcpy(prb, &(problem[0]), PROBLEM_PAIR_BYTES * num_problems, cudaMemcpyHostToDevice);
    }

    void run_main_kernel(int blocks_to_launch, int min_period, double epsilon, int max_batch_size, FILE* fptr = nullptr) {

        // run the kernel:
        launch_main_kernel(blocks_to_launch,
            ctx, prb, srb, smd, global_counters, nnue, freenodes, hrb,
            prb_size, srb_size, hrb_size,
            max_width, max_height, max_pop, rollout_gens,
            min_period, epsilon
        );

        if (fptr != nullptr) {
            // extract training data into contiguous gmem:
            makennuedata<<<blocks_to_launch / 2, 32>>>(
                prb, freenodes, global_counters, dataset, prb_size, drb_size
            );
        }

        enheap_then_deheap(hrb, global_counters, heap, hrb_size, max_batch_size >> 12, freenodes, prb_size);

        cudaMemcpy(host_counters, global_counters, 512, cudaMemcpyDeviceToHost);

        if (fptr != nullptr) {
            if (host_counters[COUNTER_READING_HEAD] >= drb_hwm + 2 * drb_size) {

                std::cout << "reading head position: " << host_counters[COUNTER_READING_HEAD] << std::endl;

                // we have a fresh batch of training data:
                cudaMemcpy(host_dataset, dataset, 32 * drb_size, cudaMemcpyDeviceToHost);

                fwrite(host_dataset, 32, drb_size, fptr);

                // update high water mark:
                drb_hwm = host_counters[COUNTER_READING_HEAD];
            }
        }
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

    // proportion of time in which we make random decisions:
    double epsilon = 0.25;

    kc::ProblemHolder ph("examples/eater.rle");
    auto problem = ph.swizzle_problem();
    auto stator = ph.swizzle_stator();

    SilkGPU silk(524288, 16384, 1048576);

    {
        uint4* nnue_h;
        cudaMallocHost((void**) &nnue_h, NNUE_BYTES);
        // load NNUE:
        FILE *fptr = fopen("nnue/nnue_399M.dat", "r");
        fread(nnue_h, 512, 7473, fptr);
        fclose(fptr);
        cudaMemcpy(silk.nnue, nnue_h, NNUE_BYTES, cudaMemcpyHostToDevice);
        cudaFreeHost(nnue_h);
    }

    silk.inject_problem(problem, stator);

    // FILE* fptr = fopen("dataset.bin", "w");

    while (true) {
        int problems = silk.host_counters[COUNTER_MIDDLE_HEAD] - silk.host_counters[COUNTER_READING_HEAD];
        int total_problems = silk.host_counters[COUNTER_WRITING_HEAD] - silk.host_counters[COUNTER_READING_HEAD];

        int lower_batch_size = 4096;
        int upper_batch_size = (silk.prb_size >> 4) - 4096;
        int medium_batch_size = ((3 * silk.prb_size) >> 5) - (total_problems >> 3);

        int batch_size = hh::max(lower_batch_size, hh::min(medium_batch_size, upper_batch_size));
        batch_size &= 0x7ffff000;

        std::cout << "Open problems: \033[31;1m" << total_problems << "\033[0m; batch size: \033[32;1m" << batch_size << "\033[0m" << std::endl;
        silk.run_main_kernel(problems, 9999, epsilon, batch_size);
        for (int i = 0; i < 64; i++) {
            std::cout << silk.host_counters[i] << " ";
        }
        std::cout << std::endl;
        if (problems == 0) { break; }
    }

    // fclose(fptr);

    uint64_t solcount = silk.host_counters[COUNTER_SOLUTION_HEAD];

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
