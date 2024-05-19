#include <silk/mainloop.hpp>

#define COUNTER_READING_HEAD 32
#define COUNTER_WRITING_HEAD 33

__global__ void computecellorbackup(
        uint4* ctx, // common context for all problems
        uint4* prb, // problem ring buffer
        uint32_t prb_size,
        uint64_t* global_counters,
        int max_width, int max_height, int max_pop, int gens, int min_period
    ) {

    // determine which problem to load:
    uint32_t logical_block_idx = 0;
    if (threadIdx.x == 0) {
        logical_block_idx = hh::atomic_add(global_counters + COUNTER_READING_HEAD, 2) & (prb_size - 1);
    }
    logical_block_idx = hh::shuffle_32(logical_block_idx, 0);
    const uint4* reading_location = prb + 352 * ((uint64_t) (logical_block_idx >> 1));
    uint32_t block_parity = logical_block_idx & 1;

    // load problem metadata:
    uint4 metadata = reading_location[threadIdx.x + 320];

    if (hh::shuffle_32(metadata.y, block_parity) == 0) {
        // we store problems in pairs for memory compression; however,
        // initially we only inject one problem so we want the ability
        // to have elements of the pair early-exit:
        return;
    }

    // ***** beginning of serious kernel *****

    // extract relevant parts of metadata:
    uint32_t px = hh::shuffle_32(metadata.y, 2);
    uint32_t py = hh::shuffle_32(metadata.y, 3);
    uint32_t perturbation = metadata.x;

    // initialise shared memory:
    __shared__ uint32_t smem[384];
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        smem[32*i + threadIdx.x] = 0;
    }
    __syncthreads();

    uint4 ad0, ad1, ad2, al2, al3, ad4, ad5, ad6;

    kc::load4(reading_location,       ad0.y, ad1.y, ad2.y, al2.y);
    kc::load4(reading_location + 32,  al3.y, ad4.y, ad5.y, ad6.y);
    kc::load4(reading_location + 64,  ad0.z, ad1.z, ad2.z, al2.z);
    kc::load4(reading_location + 96,  al3.z, ad4.z, ad5.z, ad6.z);
    kc::load4(reading_location + 128, ad0.w, ad1.w, ad2.w, al2.w);
    kc::load4(reading_location + 160, al3.w, ad4.w, ad5.w, ad6.w);
    kc::load4(reading_location + 192 + block_parity * 64, ad0.x, ad1.x, ad2.x, al2.x);
    kc::load4(reading_location + 224 + block_parity * 64, al3.x, ad4.x, ad5.x, ad6.x);

    uint4 stator = ctx[threadIdx.x];

    int return_code = kc::mainloop(
        ad0, ad1, ad2, al2, al3, ad4, ad5, ad6, stator,
        perturbation, px, py, max_width, max_height, max_pop, gens,
        smem, smem + 352
    );

    if ((return_code >= 1) && (return_code < min_period)) {
        // we have found a restabilisation or oscillator of period
        // lower than the threshold, so this does not count as a
        // solution; modify the return code accordingly:
        return_code = -1;
    }

    // flush metrics:
    __syncthreads();
    hh::atomic_add(global_counters + threadIdx.x, smem[threadIdx.x + 352]);

    if (return_code == -1) {
        // dead-end: uninteresting result or contradiction:
        return;
    } else if (return_code >= 0) {
        // we have found a solution:
        // TODO save solution
        return;
    }

    // ***** beginning of hard branch handling *****
    
}
