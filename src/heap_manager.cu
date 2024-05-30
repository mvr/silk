#include "common.hpp"
#include <silk/heap.hpp>


__global__ void __launch_bounds__(1024, 1) enheap(const uint64_t* hrb, uint64_t* global_counters, uint4* heap, int hrb_size) {

    __shared__ uint64_t smem[2048];

    // load from global counters:
    uint64_t hrb_start = global_counters[COUNTER_HRB_READING_HEAD];
    uint64_t hrb_end   = global_counters[COUNTER_HRB_WRITING_HEAD];
    int heap_elements  = global_counters[COUNTER_HEAP_ELEMENTS];

    int things_to_enqueue = (hrb_end - hrb_start) >> 11;

    if (things_to_enqueue == 0) { return; }

    for (int i = 0; i < things_to_enqueue; i++) {

        hh::vec<uint64_t, 2> x;
        int offset = (hrb_start + threadIdx.x) & (hrb_size - 1);
        x[0] = hrb[offset];
        x[1] = hrb[offset + 1024];

        // this contains a __syncthreads() call:
        kc::heap_parallel_insert<5>(x, heap_elements+1, heap, smem);

        hrb_start += 2048;
        heap_elements += 1;
    }

    // update global counters:
    global_counters[COUNTER_HRB_READING_HEAD] = hrb_start;
    global_counters[COUNTER_HEAP_ELEMENTS] = heap_elements;
}


__global__ void __launch_bounds__(1024, 1) deheap(const uint64_t* hrb, uint64_t* global_counters, uint4* heap, int hrb_size, int max_elements, uint32_t* free_nodes, int prb_size) {

    __shared__ uint64_t smem[2048];
    uint64_t hrb_start = global_counters[COUNTER_HRB_READING_HEAD];
    uint64_t hrb_end   = global_counters[COUNTER_HRB_WRITING_HEAD];
    int heap_elements  = global_counters[COUNTER_HEAP_ELEMENTS];
    uint64_t middle_head = global_counters[COUNTER_MIDDLE_HEAD];

    if (threadIdx.x == 0) {
        global_counters[COUNTER_READING_HEAD] = middle_head;
    }

    middle_head >>= 1;

    int extra_values = hrb_end - hrb_start;
    int things_to_remove = hh::min(max_elements, heap_elements);

    uint32_t prb_mask = (prb_size >> 1) - 1;

    for (int i = 0; i < things_to_remove; i++) {
        hh::vec<uint64_t, 2> x = kc::load_heap_element<5>(1, heap);
        free_nodes[(middle_head + threadIdx.x) & prb_mask] = x[0];
        free_nodes[(middle_head + threadIdx.x + 1024) & prb_mask] = x[1];
        kc::heap_parallel_delete<5>(heap_elements-i, heap, smem);
        middle_head += 2048;
    }

    heap_elements -= things_to_remove;

    __syncthreads();

    if (things_to_remove < max_elements) {
        hh::vec<uint64_t, 2> x;
        int offset = (hrb_start + threadIdx.x) & (hrb_size - 1);
        x[0] = hrb[offset];
        x[1] = hrb[offset + 1024];
        if (threadIdx.x < extra_values) {
            free_nodes[(middle_head + threadIdx.x) & prb_mask] = x[0];
        }
        if (threadIdx.x + 1024 < extra_values) {
            free_nodes[(middle_head + threadIdx.x + 1024) & prb_mask] = x[1];
        }
        middle_head += extra_values;
        global_counters[COUNTER_HRB_READING_HEAD] = 0;
        global_counters[COUNTER_HRB_WRITING_HEAD] = 0;
    }

    global_counters[COUNTER_HEAP_ELEMENTS] = heap_elements;
    global_counters[COUNTER_MIDDLE_HEAD] = middle_head << 1;

}

void enheap_then_deheap(const uint64_t* hrb, uint64_t* global_counters, uint4* heap, int hrb_size, int max_elements, uint32_t* free_nodes, int prb_size) {

    enheap<<<1, 1024>>>(hrb, global_counters, heap, hrb_size);
    deheap<<<1, 1024>>>(hrb, global_counters, heap, hrb_size, max_elements, free_nodes, prb_size);

}
