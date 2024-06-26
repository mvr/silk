#include "cqueue.hpp"
#include <silk/perm.hpp>
#include <thread>


/**
 * Rather trivial kernel that produces training data from the
 * output of computecellorbackup().
 */
__global__ void makennuedata(const uint4* prb, const uint32_t* freenodes, const uint64_t* global_counters, uint8_t* dataset, float* predictions, uint32_t prb_size, uint32_t drb_size) {

    constexpr uint64_t uint4s_per_pp = PROBLEM_PAIR_BYTES >> 4;

    // get the location from which to read:
    uint64_t pair_idx = (global_counters[COUNTER_READING_HEAD] >> 1) + blockIdx.x;
    if (pair_idx >= (global_counters[COUNTER_MIDDLE_HEAD] >> 1)) { return; }
    uint32_t node_loc = freenodes[pair_idx & ((prb_size >> 1) - 1)];
    pair_idx &= (drb_size - 1);

    __shared__ uint32_t metadata[32];

    // load the metadata:
    const uint32_t* metadata_location = ((const uint32_t*) (prb + node_loc * uint4s_per_pp + (uint4s_per_pp - 24)));
    metadata[threadIdx.x] = metadata_location[threadIdx.x];
    __syncthreads();

    uint32_t signature = metadata_location[threadIdx.x + 64];
    float total_loss = 0.0f;
    {
        uint32_t info_0 = metadata[5];
        if (metadata[8]) {
            uint32_t info_gain = hh::min(metadata[13] - info_0, ((uint32_t) 40));
            float sub_loss = __int_as_float(metadata[12]);
            sub_loss = hh::max(0.0f, hh::min(sub_loss, 1.0f));
            total_loss += 0.375f + 0.125f * sub_loss - 0.0078125f * info_gain;
        }
        if (metadata[16]) {
            uint32_t info_gain = hh::min(metadata[21] - info_0, ((uint32_t) 40));
            float sub_loss = __int_as_float(metadata[20]);
            sub_loss = hh::max(0.0f, hh::min(sub_loss, 1.0f));
            total_loss += 0.375f + 0.125f * sub_loss - 0.0078125f * info_gain;
        }
        int32_t loss_bits = ((int32_t) (total_loss * 16777216.0f));
        loss_bits = hh::max(((int32_t) 0), hh::min(loss_bits, ((int32_t) 0xffffff)));
        if (threadIdx.x >= 29) {
            signature = (loss_bits >> ((threadIdx.x - 29) * 8)) & 255;
        }
    }

    if (threadIdx.x == 0) { total_loss = __int_as_float(metadata[4]); }

    __syncthreads();
    dataset[pair_idx * 32 + threadIdx.x] = ((uint8_t) signature);
    if (threadIdx.x < 2) { predictions[pair_idx * 2 + threadIdx.x] = total_loss; }
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
    float* predictions;
    uint32_t* freenodes;
    uint64_t* hrb;
    uint4* heap;

    // host-side pointers:
    uint64_t* host_counters;
    uint32_t* host_freenodes;
    uint64_t* host_dataset;
    float* host_predictions;
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
    bool has_data;

    SilkGPU(uint64_t prb_capacity, uint64_t srb_capacity, uint64_t drb_capacity,
            int active_width, int active_height, int active_pop) {

        has_data = false;
        uint64_t hrb_capacity = prb_capacity >> 4;
        if (hrb_capacity < 8192) { hrb_capacity = 8192; }

        cudaMalloc((void**) &ctx, 512 * 2);
        cudaMalloc((void**) &prb, (PROBLEM_PAIR_BYTES >> 1) * prb_capacity);
        cudaMalloc((void**) &srb, 4096 * srb_capacity);
        cudaMalloc((void**) &smd, 4 * srb_capacity);
        cudaMalloc((void**) &global_counters, 512);
        cudaMalloc((void**) &nnue, NNUE_BYTES);

        cudaMalloc((void**) &freenodes, 2 * prb_capacity);
        cudaMallocHost((void**) &host_freenodes, 2 * prb_capacity);
        cudaMallocHost((void**) &host_srb, 4096 * srb_capacity);
        cudaMallocHost((void**) &host_smd, 4 * srb_capacity);

        if (drb_capacity > 0) {
            cudaMalloc((void**) &dataset, 32 * drb_capacity);
            cudaMalloc((void**) &predictions, 8 * drb_capacity);
            cudaMallocHost((void**) &host_dataset, 32 * drb_capacity);
            cudaMallocHost((void**) &host_predictions, 8 * drb_capacity);
        }

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

        cudaMemset(ctx, 0, 512 * 2);
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
        cudaFree(freenodes);
        cudaFree(heap);
        cudaFree(hrb);
        cudaFreeHost(host_counters);
        cudaFreeHost(host_freenodes);
        cudaFreeHost(host_srb);
        cudaFreeHost(host_smd);

        if (drb_size > 0) {
            cudaFree(dataset);
            cudaFree(predictions);
            cudaFreeHost(host_dataset);
            cudaFreeHost(host_predictions);
        }
    }

    void inject_problem(std::vector<uint32_t> problem, std::vector<uint32_t> stator, std::vector<uint32_t> exempt) {

        int num_problems = (4 * problem.size()) / PROBLEM_PAIR_BYTES;

        host_counters[COUNTER_WRITING_HEAD] = 2 * num_problems;
        host_counters[COUNTER_MIDDLE_HEAD] = 2 * num_problems;
        drb_hwm = host_counters[COUNTER_MIDDLE_HEAD];

        cudaMemcpy(global_counters, host_counters, 512, cudaMemcpyHostToDevice);
        cudaMemcpy(ctx, &(stator[0]), 512, cudaMemcpyHostToDevice);
        cudaMemcpy(((char*)ctx)+512, &(exempt[0]), 512, cudaMemcpyHostToDevice);
        cudaMemcpy(prb, &(problem[0]), PROBLEM_PAIR_BYTES * num_problems, cudaMemcpyHostToDevice);
    }

    void run_main_kernel(int blocks_to_launch, int open_problems, int min_period, int min_stable, int max_batch_size, bool make_data, SolutionQueue* status_queue, int batches = 4) {

        // if we are generating training data, then explore more
        // (75% random + 25% NNUE); otherwise, mostly follow the
        // neural network (5% random + 95% NNUE).
        double epsilon = (make_data) ? 0.75 : 0.05;

        // for batch 0, we know exactly the number of blocks:
        int batch_size = blocks_to_launch;
        int max_open_problems = open_problems;

        // Asynchronous loop: we enqueue multiple batches to run on the GPU:
        for (int bidx = 0; bidx < batches; bidx++) {

            // run the kernel:
            launch_main_kernel(batch_size,
                ctx, prb, srb, smd, global_counters, nnue, freenodes, hrb,
                prb_size, srb_size, hrb_size,
                max_width, max_height, max_pop, min_stable, rollout_gens,
                min_period, epsilon
            );

            if (make_data) {
                // extract training data into contiguous gmem:
                makennuedata<<<batch_size / 2, 32>>>(
                    prb, freenodes, global_counters, dataset, predictions, prb_size, drb_size
                );
            }

            // select the problems to run next:
            enheap_then_deheap(hrb, global_counters, heap, hrb_size, max_batch_size >> 12, freenodes, prb_size);

            // for subsequent batches, we do not know on the host side
            // the exact number of blocks to launch, but we have an upper
            // bound so we launch this number (the excess blocks will
            // early-exit):
            max_open_problems += batch_size;
            batch_size = hh::min(max_batch_size, max_open_problems);
        }

        // This moderately expensive host-side operation will execute
        // whilst the kernels are running:
        if (has_data) {

            SolutionMessage sm; sm.message_type = MESSAGE_DATA;
            sm.nnue_data.resize(drb_size * 4);

            float target_mean = 0.0f;
            float pred_mean = 0.0f;

            // randomly shuffle training data (32-byte vectors):
            uint64_t random_key = hh::fibmix(host_counters[COUNTER_READING_HEAD]);
            for (uint32_t i = 0; i < drb_size; i++) {
                uint32_t k = kc::random_perm(i, drb_size, random_key);
                for (uint32_t j = 0; j < 4; j++) {
                    sm.nnue_data[4*i + j] = host_dataset[4*k + j];
                }
                target_mean += host_predictions[2*i+1];
                pred_mean += host_predictions[2*i];
            }

            target_mean /= ((float) drb_size);
            pred_mean /= ((float) drb_size);

            float variance = 0.0f;
            float variance_unexplained = 0.0f;

            for (uint32_t i = 0; i < drb_size; i++) {
                float target_demeaned = host_predictions[2*i+1] - target_mean;
                float residual = host_predictions[2*i+1] - host_predictions[2*i];
                variance += target_demeaned * target_demeaned;
                variance_unexplained += residual * residual;
            }

            variance /= ((float) drb_size);
            variance_unexplained /= ((float) drb_size);

            sm.floats[0] = target_mean;
            sm.floats[1] = pred_mean;
            sm.floats[2] = variance;
            sm.floats[3] = variance_unexplained;

            status_queue->enqueue(sm);
            has_data = false;
        }

        // this is synchronous, so awaits the completion of the kernels:
        cudaMemcpy(host_counters, global_counters, 512, cudaMemcpyDeviceToHost);

        if (make_data) {
            if (host_counters[COUNTER_READING_HEAD] >= drb_hwm + 2 * drb_size) {

                // we have a fresh batch of training data:
                cudaMemcpy(host_dataset, dataset, 32 * drb_size, cudaMemcpyDeviceToHost);
                cudaMemcpy(host_predictions, predictions, 8 * drb_size, cudaMemcpyDeviceToHost);

                // update high water mark:
                drb_hwm = host_counters[COUNTER_READING_HEAD];

                has_data = true;
            }
        }

        hh::reportCudaError(cudaGetLastError());
    }
};

void run_main_loop(SilkGPU &silk, const uint64_t* perturbation, SolutionQueue* status_queue, bool make_data, int min_report_period, int min_stable) {

    int open_problems = silk.host_counters[COUNTER_WRITING_HEAD] - silk.host_counters[COUNTER_READING_HEAD];
    uint64_t last_solution_count = 0;

    while (open_problems) {
        int problems = silk.host_counters[COUNTER_MIDDLE_HEAD] - silk.host_counters[COUNTER_READING_HEAD];

        int lower_batch_size = 4096;
        int upper_batch_size = (silk.prb_size >> 4) - 4096;
        int medium_batch_size = ((3 * silk.prb_size) >> 5) - (open_problems >> 3);

        int batch_size = hh::max(lower_batch_size, hh::min(medium_batch_size, upper_batch_size));
        batch_size &= 0x7ffff000;

        silk.run_main_kernel(problems, open_problems, min_report_period, min_stable, batch_size, make_data, status_queue);

        open_problems = silk.host_counters[COUNTER_WRITING_HEAD] - silk.host_counters[COUNTER_READING_HEAD];

        {
            // report to status queue:
            SolutionMessage sm; sm.message_type = MESSAGE_STATUS;
            sm.return_code = 0; // TODO put device ID here
            for (int i = 0; i < 64; i++) { sm.metrics[i] = silk.host_counters[i]; }
            sm.metrics[METRIC_PRB_SIZE] = silk.prb_size;
            sm.metrics[METRIC_BATCH_SIZE] = problems;
            status_queue->enqueue(sm);
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
            cudaMemcpy(silk.host_smd, silk.smd + starting_idx, 4 * solcount, cudaMemcpyDeviceToHost);

            for (uint64_t i = 0; i < solcount; i++) {
                SolutionMessage sm;
                sm.message_type = MESSAGE_SOLUTION;
                sm.return_code = silk.host_smd[i];
                for (uint64_t j = 0; j < 1024; j++) {
                    sm.solution[j] = silk.host_srb[1024 * i + j];
                }
                for (uint64_t j = 0; j < 64; j++) {
                    sm.perturbation[j] = perturbation[j];
                }
                status_queue->enqueue(sm);
            }
        }
    }

    {
        // signal to the status queue that we've finished:
        SolutionMessage sm;
        sm.message_type = MESSAGE_KILL_THREAD;
        status_queue->enqueue(sm);
    }
}

int silk_main(int active_width, int active_height, int active_pop, std::string input_filename, std::string nnue_filename, int num_cadical_threads, int min_report_period, int min_stable, bool exempt_existing, std::string dataset_filename) {

    // ***** LOAD PROBLEM *****

    kc::ProblemHolder ph(input_filename, exempt_existing);

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
    auto exempt = ph.swizzle_exempt();

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

    bool make_data = dataset_filename.size() > 0;

    // these probably don't need changing:
    size_t srb_capacity = 16384;
    size_t drb_capacity = make_data ? 1048576 : 0;
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

    silk.inject_problem(problem, stator, exempt);

    // ***** ESTABLISH COMMUNICATIONS *****

    SolutionQueue status_queue;
    SolutionQueue solution_queue;
    PrintQueue print_queue;

    std::thread status_thread(status_thread_loop, 1, num_cadical_threads, &status_queue, &solution_queue, &print_queue, dataset_filename);
    std::vector<std::thread> cadical_threads;
    std::thread print_thread(print_thread_loop, num_cadical_threads + 1, &print_queue);

    for (int i = 0; i < num_cadical_threads; i++) {
        cadical_threads.emplace_back(solution_thread_loop, &solution_queue, &print_queue);
    }

    run_main_loop(silk, &(ph.perturbation[0]), &status_queue, make_data, min_report_period, min_stable);

    // ***** TEAR DOWN THREADS *****

    status_thread.join();
    for (int i = 0; i < num_cadical_threads; i++) { cadical_threads[i].join(); }
    print_thread.join();

    return 0;
}
