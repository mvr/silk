#pragma once
#include <cpads/sorting/bitonic.hpp>

namespace kc {

template<size_t Log2WarpsPerBlock, bool ReplaceWithZeros=false>
_DI_ hh::vec<uint64_t, 2> load_heap_element(int n, uint4* heap) {
    constexpr uint64_t heap_stride = 32 << Log2WarpsPerBlock;
    uint4 yv = heap[(n - 1) * heap_stride + threadIdx.x];
    hh::vec<uint64_t, 2> y;
    y[0] = yv.x | (((uint64_t) yv.y) << 32);
    y[1] = yv.z | (((uint64_t) yv.w) << 32);
    if constexpr (ReplaceWithZeros) {
        yv.x = 0; yv.y = 0; yv.z = 0; yv.w = 0;
        heap[(n - 1) * heap_stride + threadIdx.x] = yv;
    }
    return y;
}

template<size_t Log2WarpsPerBlock>
_DI_ void store_heap_element(int n, uint4* heap, hh::vec<uint64_t, 2> &y) {
    constexpr uint64_t heap_stride = 32 << Log2WarpsPerBlock;
    uint4 yv;
    yv.x = y[0];
    yv.y = y[0] >> 32;
    yv.z = y[1];
    yv.w = y[1] >> 32;
    heap[(n - 1) * heap_stride + threadIdx.x] = yv;
}

template<size_t Log2WarpsPerBlock>
_DI_ void heap_parallel_insert(hh::vec<uint64_t, 2> &x, int n, uint4* heap, uint64_t *smem) {

    __syncthreads();
    hh::block_bitonic_sort<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(x, smem);
    bool x_sorted = true;

    int child_idx = n;
    while (child_idx > 1) {
        if (!x_sorted) { hh::block_bitonic_merge<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(x, smem); }
        int parent_idx = child_idx >> 1;
        auto y = load_heap_element<Log2WarpsPerBlock>(parent_idx, heap);

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            hh::compare_and_swap(y[i], x[i]);
        }

        hh::block_bitonic_merge<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(y, smem, true);
        store_heap_element<Log2WarpsPerBlock>(child_idx, heap, y);
        child_idx = parent_idx;
    }

    hh::block_bitonic_merge<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(x, smem, true);
    store_heap_element<Log2WarpsPerBlock>(1, heap, x);
    __syncthreads();
}

template<size_t Log2WarpsPerBlock>
_DI_ void heap_parallel_delete(int n, uint4* heap, uint64_t *smem) {

    constexpr int lastThread = (32 << Log2WarpsPerBlock) - 1;

    __syncthreads();
    auto x = load_heap_element<Log2WarpsPerBlock, true>(n, heap);
    __syncthreads();
    int parent_idx = 1;
    while (2 * parent_idx < n) {
        int lidx = 2 * parent_idx;
        int ridx = lidx + 1;

        auto xl = load_heap_element<Log2WarpsPerBlock>(lidx, heap);
        auto xr = load_heap_element<Log2WarpsPerBlock>(ridx, heap);

        __syncthreads();
        if (threadIdx.x == lastThread) {
            smem[0] = xl[1];
            smem[1] = xr[1];
        }
        __syncthreads();
        bool left_was_bigger = (smem[0] >= smem[1]);
        __syncthreads();

        hh::block_bitonic_merge<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(xr, smem);
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            hh::compare_and_swap(xl[i], xr[i]);
        }
        hh::block_bitonic_merge<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(xr, smem);
        hh::block_bitonic_merge<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(xl, smem, true);
        store_heap_element<Log2WarpsPerBlock>(lidx + left_was_bigger, heap, xl);

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            hh::compare_and_swap(x[i], xr[i]);
        }
        hh::block_bitonic_merge<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(xr, smem, true);
        hh::block_bitonic_merge<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(x, smem, true);
        store_heap_element<Log2WarpsPerBlock>(parent_idx, heap, xr);
        parent_idx = ridx - left_was_bigger;
    }

    store_heap_element<Log2WarpsPerBlock>(parent_idx, heap, x);
    __syncthreads();
}

} // namespace kc
