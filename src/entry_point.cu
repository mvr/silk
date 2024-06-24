#include "cqueue.hpp"
#include <silk/perm.hpp>
#include <thread>
#include <atomic>


__global__ void srb_to_prb(const uint4* srb, uint4* prb, const uint32_t* freenodes, const uint64_t* global_counters, uint32_t prb_size) {

    constexpr uint64_t uint4s_per_pp = PROBLEM_PAIR_BYTES >> 4;

    uint64_t pair_idx = (global_counters[COUNTER_READING_HEAD] >> 1) + blockIdx.x;
    uint32_t node_loc = freenodes[pair_idx & ((prb_size >> 1) - 1)];

    const uint32_t* reading_location = ((const uint32_t*) (srb + blockIdx.x * uint4s_per_pp));
    uint32_t* writing_location = ((uint32_t*) (prb + node_loc * uint4s_per_pp));

    #pragma unroll
    for (int i = 0; i < 5; i++) {
        writing_location[224 * i + threadIdx.x] = reading_location[224 * i + threadIdx.x];
    }
}


__global__ void prb_to_srb(const uint4* prb, uint4* srb, const uint32_t* freenodes, const uint64_t* global_counters, uint32_t prb_size, uint64_t multiplier, uint64_t offset, uint64_t modulus) {

    constexpr uint64_t uint4s_per_pp = PROBLEM_PAIR_BYTES >> 4;

    uint64_t mangled_idx = (multiplier * (blockIdx.x + offset)) % modulus;
    uint64_t pair_idx = (global_counters[COUNTER_READING_HEAD] >> 1) + mangled_idx;
    uint32_t node_loc = freenodes[pair_idx & ((prb_size >> 1) - 1)];

    const uint32_t* reading_location = ((const uint32_t*) (prb + node_loc * uint4s_per_pp));
    uint32_t* writing_location = ((uint32_t*) (srb + blockIdx.x * uint4s_per_pp));

    #pragma unroll
    for (int i = 0; i < 5; i++) {
        writing_location[224 * i + threadIdx.x] = reading_location[224 * i + threadIdx.x];
    }
}


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
    uint64_t last_solution_count;

    uint64_t perturbation[64];
    bool has_data;

    SilkGPU(uint64_t prb_capacity, uint64_t srb_capacity, uint64_t drb_capacity,
            int active_width, int active_height, int active_pop) {

        has_data = false;
        uint64_t hrb_capacity = prb_capacity >> 4;
        if (hrb_capacity < 8192) { hrb_capacity = 8192; }

        cudaMalloc((void**) &ctx, 512);
        cudaMalloc((void**) &prb, (PROBLEM_PAIR_BYTES >> 1) * prb_capacity);
        cudaMalloc((void**) &srb, 4096 * srb_capacity);
        cudaMalloc((void**) &smd, 4 * srb_capacity);
        cudaMalloc((void**) &global_counters, 512);

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

        cudaMemset(ctx, 0, 512);

        max_width = active_width;
        max_height = active_height;
        max_pop = active_pop;
        rollout_gens = 6;
        drb_hwm = 0;
        last_solution_count = 0;
    }

    ~SilkGPU() {
        cudaFree(ctx);
        cudaFree(prb);
        cudaFree(srb);
        cudaFree(smd);
        cudaFree(global_counters);
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

    void set_stator(std::vector<uint32_t> stator) {
        cudaMemcpy(ctx, &(stator[0]), 512, cudaMemcpyHostToDevice);
    }

    void inject_problems(std::vector<uint32_t> problem) {

        int num_problem_pairs = (4 * problem.size()) / PROBLEM_PAIR_BYTES;

        host_counters[COUNTER_WRITING_HEAD] = host_counters[COUNTER_READING_HEAD] + 2 * num_problem_pairs;
        host_counters[COUNTER_MIDDLE_HEAD] = host_counters[COUNTER_WRITING_HEAD];
        drb_hwm = host_counters[COUNTER_MIDDLE_HEAD];

        cudaMemcpy(global_counters, host_counters, 512, cudaMemcpyHostToDevice);
        if (num_problem_pairs > 0) {
            cudaMemcpy(srb, &(problem[0]), PROBLEM_PAIR_BYTES * num_problem_pairs, cudaMemcpyHostToDevice);
            srb_to_prb<<<num_problem_pairs, 224>>>(srb, prb, freenodes, global_counters, prb_size);
        }
    }

    void split(uint64_t problem_pairs, uint64_t num_batches, ProblemQueue* master_queue) {

        uint64_t multiplier = ((uint64_t) (((double) problem_pairs) * 0.38196601125)) | 1ull;
        while (hh::binary_gcd(multiplier, problem_pairs) > 1) { multiplier += 2; }

        for (uint64_t i = 0; i < num_batches; i++) {
            uint64_t offset = (i * problem_pairs) / num_batches;
            uint64_t offset2 = ((i+1) * problem_pairs) / num_batches;
            uint64_t bsize = offset2 - offset;
            prb_to_srb<<<bsize, 224>>>(prb, srb, freenodes, global_counters, prb_size, multiplier, offset, problem_pairs);
            std::vector<uint32_t> problem((PROBLEM_PAIR_BYTES >> 2) * bsize);
            cudaMemcpy(&(problem[0]), srb, PROBLEM_PAIR_BYTES * bsize, cudaMemcpyDeviceToHost);
            ProblemMessage pm;
            pm.message_type = MESSAGE_PROBLEMS;
            pm.problem_data = problem;
            master_queue->enqueue(pm);
        }
    }

    void run_main_kernel(const float4* nnue, int blocks_to_launch, int open_problems, int min_period, int min_stable, int max_batch_size, bool make_data, SolutionQueue* status_queue, int batches = 4) {

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

void run_main_loop(int stream_id, const float4* nnue, SilkGPU &silk, const uint64_t* perturbation, SolutionQueue* status_queue, bool make_data, int min_report_period, int min_stable, std::atomic<int64_t> *approx_batches, ProblemQueue *master_queue) {

    int elapsed_iters = 0;
    int open_problems = silk.host_counters[COUNTER_WRITING_HEAD] - silk.host_counters[COUNTER_READING_HEAD];

    approx_batches->fetch_add(-1);

    while (open_problems) {
        int problems = silk.host_counters[COUNTER_MIDDLE_HEAD] - silk.host_counters[COUNTER_READING_HEAD];

        int lower_batch_size = 4096;
        int upper_batch_size = (silk.prb_size >> 4) - 4096;
        int medium_batch_size = ((3 * silk.prb_size) >> 5) - (open_problems >> 3);

        int batch_size = hh::max(lower_batch_size, hh::min(medium_batch_size, upper_batch_size));
        batch_size &= 0x7ffff000;

        silk.run_main_kernel(nnue, problems, open_problems, min_report_period, min_stable, batch_size, make_data, status_queue);

        open_problems = silk.host_counters[COUNTER_WRITING_HEAD] - silk.host_counters[COUNTER_READING_HEAD];

        {
            // report to status queue:
            SolutionMessage sm; sm.message_type = MESSAGE_STATUS;
            sm.return_code = stream_id; // identify origin of status message:
            for (int i = 0; i < 64; i++) { sm.metrics[i] = silk.host_counters[i]; }
            sm.metrics[METRIC_PRB_SIZE] = silk.prb_size;
            sm.metrics[METRIC_BATCH_SIZE] = problems;
            status_queue->enqueue(sm);
        }

        while (silk.host_counters[COUNTER_SOLUTION_HEAD] > silk.last_solution_count) {
            uint64_t next_solution_count = silk.host_counters[COUNTER_SOLUTION_HEAD];
            if ((next_solution_count / silk.srb_size) > (silk.last_solution_count / silk.srb_size)) {
                next_solution_count = ((silk.last_solution_count / silk.srb_size) + 1) * silk.srb_size;
            }
            uint64_t solcount = next_solution_count - silk.last_solution_count;
            uint64_t starting_idx = silk.last_solution_count % silk.srb_size;

            silk.last_solution_count = next_solution_count;

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

        elapsed_iters += 1;
        if ((elapsed_iters >= 50) && (open_problems >= 16384) && (approx_batches->load() <= 20)) {
            // we should dump the search progress into the queue:
            int new_batches = (open_problems + 8191) >> 13;
            if (new_batches < 4) { new_batches = 4; }
            approx_batches->fetch_add(new_batches);

            // empty heap:
            enheap_then_deheap(silk.hrb, silk.global_counters, silk.heap, silk.hrb_size, -1, silk.freenodes, silk.prb_size);
            cudaMemcpy(silk.host_counters, silk.global_counters, 512, cudaMemcpyDeviceToHost);

            // make batches:
            uint64_t problem_pairs = open_problems >> 1;
            silk.split(problem_pairs, new_batches, master_queue);
            break;
        }
    }
}

void gpu_thread_loop(ProblemQueue *problem_queue, ProblemQueue *master_queue, SolutionQueue *status_queue,
    int stream_id, int device_id, size_t prb_capacity, const float4* nnue, bool make_data, int active_width,
    int active_height, int active_pop, const kc::ProblemHolder *ph, int min_report_period, int min_stable,
    std::atomic<int64_t> *approx_batches) {

    cudaSetDevice(device_id);

    auto stator = ph->swizzle_stator();

    // these probably don't need changing:
    size_t srb_capacity = 16384;
    size_t drb_capacity = make_data ? 1048576 : 0;

    SilkGPU silk(prb_capacity, srb_capacity, drb_capacity, active_width, active_height, active_pop);

    silk.set_stator(stator);

    ProblemMessage item;

    while (true) {
        problem_queue->wait_dequeue(item);
        if (item.message_type == MESSAGE_KILL_THREAD) { break; }
        silk.inject_problems(item.problem_data);
        run_main_loop(stream_id, nnue, silk, &(ph->perturbation[0]), status_queue, make_data, min_report_period, min_stable, approx_batches, master_queue);

        {
            // tell master thread that we've finished a batch:
            ProblemMessage pm;
            pm.message_type = MESSAGE_COMPLETED;
            master_queue->enqueue(pm);
        }
    }

    {
        // signal to the status queue that we've finished:
        SolutionMessage sm;
        sm.message_type = MESSAGE_KILL_THREAD;
        status_queue->enqueue(sm);
    }
}

int silk_main(int active_width, int active_height, int active_pop, std::string input_filename, std::string nnue_filename, int num_cadical_threads, int min_report_period, int min_stable, std::string dataset_filename) {

    #define REPORT_EXIT(X) if (hh::reportCudaError(X)) { std::cerr << "Error: Silk aborting due to irrecoverable GPU error." << std::endl; return 1; }

    // ***** LOAD PROBLEM *****

    kc::ProblemHolder ph(input_filename);

    int ppc = 0;
    for (int i = 0; i < 64; i++) { ppc += hh::popc64(ph.perturbation[i]); }

    if (ppc == 0) {
        std::cerr << "Error: pattern file has zero live cells in initial perturbation." << std::endl;
        return 1;
    } else {
        std::cerr << "Info: initial perturbation has " << ppc << " cells." << std::endl;
    }

    // ***** CHECK CUDA IS WORKING CORRECTLY *****

    int num_devices = 0;

    REPORT_EXIT(cudaGetDeviceCount(&num_devices))

    if (num_devices <= 0) {
        std::cerr << "Error: no CUDA-capable GPUs were detected." << std::endl;
        return 1;
    }

    std::vector<std::pair<uint64_t, float4*>> device_infos;
    int num_streams = 0;

    // ***** LOAD NNUE AND COPY TO DEVICES *****
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

        std::cerr << "Info: probing " << num_devices << " devices..." << std::endl;

        for (int i = 0; i < num_devices; i++) {
            REPORT_EXIT(cudaSetDevice(i))
            size_t free_mem = 0;
            size_t total_mem = 0;
            REPORT_EXIT(cudaMemGetInfo(&free_mem, &total_mem))
            free_mem >>= 20;
            total_mem >>= 20;
            std::cerr << "    -- device " << i << " has " << free_mem << " MiB free and " << total_mem << " MiB total." << std::endl;
            if (free_mem >= 256) {
                float4* nnue_d;
                REPORT_EXIT(cudaMalloc((void**) &nnue_d, NNUE_BYTES))
                REPORT_EXIT(cudaMemcpy(nnue_d, nnue_h, NNUE_BYTES, cudaMemcpyHostToDevice))
                device_infos.emplace_back((free_mem << 32) | ((uint64_t) i), nnue_d);
                num_streams += (free_mem >= 512) ? 2 : 1;
            }
            REPORT_EXIT(cudaDeviceSynchronize())
        }

        cudaFreeHost(nnue_h);
    }

    if (device_infos.size() == 0) {
        std::cerr << "Error: no devices were found with sufficient memory." << std::endl;
        return 1;
    }

    // sort descending, so that the devices with greater memory come first:
    std::sort(device_infos.rbegin(), device_infos.rend());

    bool make_data = dataset_filename.size() > 0;

    // ***** ESTABLISH COMMUNICATIONS *****

    SolutionQueue status_queue;
    SolutionQueue solution_queue;
    ProblemQueue problem_queue;
    ProblemQueue master_queue;
    PrintQueue print_queue;

    std::vector<std::thread> gpu_threads;

    std::atomic<int64_t> approx_batches = 1;

    std::cerr << "Info: creating " << num_streams << " streams..." << std::endl;

    for (auto&& x : device_infos) {
        size_t free_megabytes = x.first >> 32;
        std::vector<size_t> stream_capacities(2);

        if (free_megabytes < 512) {
            stream_capacities[0] = 65536;
            stream_capacities[1] = 0;
        } else {
            free_megabytes -= 224;
            stream_capacities[0] = 8192 << hh::constexpr_log2(free_megabytes / 27);
            stream_capacities[1] = 4096 << hh::constexpr_log2(free_megabytes / 18);
        }

        for (auto&& prb_capacity : stream_capacities) {
            if (prb_capacity > 0) {
                int stream_id = gpu_threads.size();
                int device_id = ((uint32_t) x.first);
                const float4* nnue = x.second;
                std::cerr << "    -- creating stream " << stream_id << " on device " << device_id << " with ring buffer size " << prb_capacity << std::endl;
                gpu_threads.emplace_back(gpu_thread_loop, &problem_queue, &master_queue, &status_queue, stream_id, device_id, prb_capacity, nnue, make_data,
                                        active_width, active_height, active_pop, &ph, min_report_period, min_stable, &approx_batches);
            }
        }
    }

    std::thread status_thread(status_thread_loop, num_streams, num_cadical_threads, &status_queue, &solution_queue, &print_queue, dataset_filename);
    std::vector<std::thread> cadical_threads;
    std::thread print_thread(print_thread_loop, num_cadical_threads + 1, &print_queue);

    for (int i = 0; i < num_cadical_threads; i++) {
        cadical_threads.emplace_back(solution_thread_loop, &solution_queue, &print_queue);
    }

    // ***** INJECT PROBLEM *****

    std::cerr << "Info: all comms established; commencing search..." << std::endl;
    {
        ProblemMessage pm;
        pm.message_type = MESSAGE_PROBLEMS;
        pm.problem_data = ph.swizzle_problem();
        problem_queue.enqueue(pm);
    }
    uint64_t batches_in_flight = 1;

    // ***** AWAIT COMPLETION *****

    while (batches_in_flight) {
        ProblemMessage item;
        master_queue.wait_dequeue(item);

        if (item.message_type == MESSAGE_COMPLETED) {
            batches_in_flight -= 1;
        } else if (item.message_type == MESSAGE_PROBLEMS) {
            ProblemMessage pm;
            pm.message_type = item.message_type;
            pm.problem_data = item.problem_data;
            problem_queue.enqueue(pm);
            batches_in_flight += 1;
        }
    }

    // ***** TEAR DOWN THREADS *****

    for (int i = 0; i < num_streams; i++) { ProblemMessage pm; pm.message_type = MESSAGE_KILL_THREAD; problem_queue.enqueue(pm); }
    for (int i = 0; i < num_streams; i++) { gpu_threads[i].join(); }

    for (auto&& x : device_infos) {
        REPORT_EXIT(cudaSetDevice(((uint32_t) x.first)))
        REPORT_EXIT(cudaDeviceSynchronize())
        REPORT_EXIT(cudaFree(x.second))
    }

    status_thread.join();
    for (int i = 0; i < num_cadical_threads; i++) { cadical_threads[i].join(); }
    print_thread.join();

    return 0;

    #undef REPORT_EXIT
}
