#include "cqueue.hpp"
#include <stdio.h>
#include <chrono>
#include <cstdarg>
#include <thread>

std::string format_string(const char* format, ...) {
    // Start with variadic arguments
    va_list args;
    va_start(args, format);

    // Determine the size needed to hold the formatted string
    int size = vsnprintf(nullptr, 0, format, args);
    va_end(args);

    // Create a buffer of the required size
    std::vector<char> buffer(size + 1); // +1 for null terminator

    // Print to the buffer
    va_start(args, format);
    vsnprintf(buffer.data(), buffer.size(), format, args);
    va_end(args);

    // Create a std::string from the buffer
    return std::string(buffer.data());
}

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
    uint32_t* host_srb;
    int32_t* host_smd;

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

    uint64_t perturbation[64];

    SilkGPU(uint64_t prb_capacity, uint64_t srb_capacity, uint64_t drb_capacity,
            int active_width, int active_height, int active_pop) {

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
        cudaMallocHost((void**) &host_srb, 4096 * srb_capacity);
        cudaMallocHost((void**) &host_smd, 4 * srb_capacity);

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

        max_width = active_width;
        max_height = active_height;
        max_pop = active_pop;
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
        cudaFreeHost(host_srb);
        cudaFreeHost(host_smd);
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

    void run_main_kernel(int blocks_to_launch, int min_period, int max_batch_size, FILE* fptr = nullptr) {

        // if we are generating training data, then explore more
        // (75% random + 25% NNUE); otherwise, mostly follow the
        // neural network (5% random + 95% NNUE).
        double epsilon = (fptr == nullptr) ? 0.05 : 0.75;

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

void run_main_loop(SilkGPU &silk, const uint64_t* perturbation, SolutionQueue* solution_queue, PrintQueue* print_queue, FILE* fptr) {

    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = t0;
    auto t2 = t0;

    uint64_t last_problems = silk.host_counters[COUNTER_READING_HEAD];

    int open_problems = silk.host_counters[COUNTER_WRITING_HEAD] - silk.host_counters[COUNTER_READING_HEAD];

    uint64_t next_elapsed_secs = 1;

    uint64_t last_solution_count = 0;

    while (open_problems) {
        int problems = silk.host_counters[COUNTER_MIDDLE_HEAD] - silk.host_counters[COUNTER_READING_HEAD];

        int lower_batch_size = 4096;
        int upper_batch_size = (silk.prb_size >> 4) - 4096;
        int medium_batch_size = ((3 * silk.prb_size) >> 5) - (open_problems >> 3);

        int batch_size = hh::max(lower_batch_size, hh::min(medium_batch_size, upper_batch_size));
        batch_size &= 0x7ffff000;

        silk.run_main_kernel(problems, 9999, batch_size, fptr);

        t2 = std::chrono::high_resolution_clock::now();

        auto total_usecs = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();
        open_problems = silk.host_counters[COUNTER_WRITING_HEAD] - silk.host_counters[COUNTER_READING_HEAD];

        uint64_t elapsed_secs = total_usecs / 1000000;

        if ((elapsed_secs == 0) || (open_problems == 0) || (elapsed_secs >= next_elapsed_secs)) {

            auto usecs = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            double total_mprobs_per_sec    = ((double) silk.host_counters[COUNTER_READING_HEAD]) / ((double) total_usecs);
            double current_mprobs_per_sec  = ((double) (silk.host_counters[COUNTER_READING_HEAD] - last_problems)) / ((double) usecs);

            t1 = t2;
            last_problems = silk.host_counters[COUNTER_READING_HEAD];

            char time_specifier;
            uint64_t denom;
            uint64_t elapsed_increment;

            if (elapsed_secs >= 1209600) { // 336 hours --> 14 days
                time_specifier = 'd';
                denom = 86400;
                elapsed_increment = 86400;
            } else if (elapsed_secs >= 28800) { // 480 minutes --> 8 hours
                time_specifier = 'h';
                denom = 3600;
                elapsed_increment = (elapsed_secs >= 172800) ? 14400 : 3600;
            } else if (elapsed_secs >= 600) { // 600 seconds --> 10 minutes
                time_specifier = 'm';
                denom = 60;
                elapsed_increment = (elapsed_secs >= 7200) ? 900 : ((elapsed_secs >= 1800) ? 300 : 60);
            } else {
                time_specifier = 's';
                denom = 1;
                elapsed_increment = (elapsed_secs >= 60) ? 30 : ((elapsed_secs >= 10) ? 5 : 1);
            }

            PrintMessage pm; pm.message_type = MESSAGE_STATUS_STRING;

            if ((elapsed_secs == 0) || (open_problems == 0)) {
                double elapsed_time = ((double) total_usecs) / (1.0e6 * ((double) denom));
                pm.contents += format_string("|%6.2f %c |", elapsed_time, time_specifier);
            } else {
                next_elapsed_secs += elapsed_increment;
                unsigned long long elapsed_time = elapsed_secs / denom;
                pm.contents += format_string("| %4d %c  |", elapsed_time, time_specifier);
            }

            pm.contents += format_string("%14llu | %8llu (%5.2f%%) | %7llu | %7.3f | %7.3f | %7.3f | %7llu | %7llu |",
                ((unsigned long long) last_problems),
                ((unsigned long long) open_problems),
                (100.0 * open_problems / silk.prb_size),
                ((unsigned long long) problems),
                ((double) (silk.host_counters[METRIC_ROLLOUT])) / last_problems,
                current_mprobs_per_sec,
                total_mprobs_per_sec,
                ((unsigned long long) (silk.host_counters[COUNTER_SOLUTION_HEAD] - silk.host_counters[METRIC_FIZZLE])),
                ((unsigned long long) silk.host_counters[METRIC_FIZZLE])
            );

            print_queue->enqueue(pm);
        }

        while (silk.host_counters[COUNTER_SOLUTION_HEAD] > last_solution_count) {
            uint64_t next_solution_count = silk.host_counters[COUNTER_SOLUTION_HEAD];
            if ((next_solution_count / silk.srb_size) > (last_solution_count / silk.srb_size)) {
                next_solution_count = ((last_solution_count / silk.srb_size) + 1) * silk.srb_size;
            }
            uint64_t solcount = next_solution_count - last_solution_count;
            uint64_t starting_idx = last_solution_count % silk.srb_size;

            last_solution_count = next_solution_count;

            cudaMemcpy(silk.host_srb, silk.srb + 256 * starting_idx, 4096 * solcount, cudaMemcpyDeviceToHost);
            cudaMemcpy(silk.host_smd, silk.smd + 4 * starting_idx, 4 * solcount, cudaMemcpyDeviceToHost);

            for (uint64_t i = 0; i < solcount; i++) {
                SolutionMessage sm;
                sm.message_type = MESSAGE_SOLUTION_STRING;
                sm.return_code = silk.host_smd[i];
                for (uint64_t j = 0; j < 1024; j++) {
                    sm.solution[j] = silk.host_srb[1024 * i + j];
                }
                for (uint64_t j = 0; j < 64; j++) {
                    sm.perturbation[j] = perturbation[j];
                }
                solution_queue->enqueue(sm);
            }
        }
    }
}

int silk_main(int active_width, int active_height, int active_pop, std::string input_filename, std::string nnue_filename, int num_cadical_threads) {

    // ***** LOAD PROBLEM *****

    kc::ProblemHolder ph("examples/eater.rle");

    int ppc = 0;
    for (int i = 0; i < 64; i++) { ppc += hh::popc64(ph.perturbation[i]); }

    if (ppc == 0) {
        std::cerr << "Error: pattern file has zero live cells in initial perturbation." << std::endl;
        return 1;
    } else {
        std::cerr << "Info: initial perturbation has " << ppc << " cells." << std::endl;
    }

    auto problem = ph.swizzle_problem();
    auto stator = ph.swizzle_stator();

    // ***** CHECK CUDA IS WORKING CORRECTLY *****

    size_t free_mem = 0;
    size_t total_mem = 0;

    if (hh::reportCudaError(cudaMemGetInfo(&free_mem, &total_mem))) {
        std::cerr << "Error: Silk aborting due to irrecoverable GPU error." << std::endl;
        return 1;
    }

    std::cerr << "Info: GPU memory statistics: " << free_mem << " bytes free; " << total_mem << " bytes total." << std::endl;

    if (free_mem < ((size_t) (1 << 28))) {
        std::cerr << "Error: Silk requires at least 256 MiB of free GPU memory to run correctly." << std::endl;
        return 1;
    }

    // these probably don't need changing:
    size_t srb_capacity = 16384;
    size_t drb_capacity = 1048576;
    free_mem -= (srb_capacity + 4096) * 4096;
    free_mem -= drb_capacity * 32;

    // calculate maximum prb_capacity that fits in free memory:
    size_t prb_capacity = free_mem / 2304;
    prb_capacity = 1 << hh::constexpr_log2(prb_capacity);

    std::cerr << "Info: allocating ring buffer to accommodate " << prb_capacity << " open problems." << std::endl;

    SilkGPU silk(prb_capacity, srb_capacity, drb_capacity, active_width, active_height, active_pop);

    {
        uint4* nnue_h;
        cudaMallocHost((void**) &nnue_h, NNUE_BYTES);
        // load NNUE:
        FILE *fptr = fopen(nnue_filename.c_str(), "r");

        if (fptr == nullptr) {
            std::cerr << "Error: failed to load NNUE from file " << nnue_filename << std::endl;
            return 1;
        }

        fread(nnue_h, 512, 7473, fptr);
        fclose(fptr);
        cudaMemcpy(silk.nnue, nnue_h, NNUE_BYTES, cudaMemcpyHostToDevice);
        cudaFreeHost(nnue_h);
    }

    silk.inject_problem(problem, stator);

    SolutionQueue solution_queue;
    PrintQueue print_queue;

    std::thread print_thread(print_thread_loop, &print_queue);
    std::vector<std::thread> cadical_threads;

    for (int i = 0; i < num_cadical_threads; i++) {
        cadical_threads.emplace_back(solution_thread_loop, &solution_queue, &print_queue);
    }

    FILE* fptr = nullptr;
    // fptr = fopen("dataset.bin", "w");

    run_main_loop(silk, &(ph.perturbation[0]), &solution_queue, &print_queue, fptr);

    if (fptr != nullptr) { fclose(fptr); }

    // ***** TEAR DOWN THREADS *****

    for (int i = 0; i < num_cadical_threads; i++) {
        SolutionMessage sm;
        sm.message_type = MESSAGE_KILL_THREAD;
        solution_queue.enqueue(sm);
    }

    for (int i = 0; i < num_cadical_threads; i++) {
        cadical_threads[i].join();
    }

    {
        PrintMessage pm;
        pm.message_type = MESSAGE_KILL_THREAD;
        print_queue.enqueue(pm);
        print_thread.join();
    }

    return 0;
}
