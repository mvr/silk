#include <silk/heap.hpp>
#include <cpads/random/prng.hpp>
#include <gtest/gtest.h>
#include <algorithm>

__global__ void copy_to_heap(const uint64_t *input, uint4* heap, int num_vecs) {

    __shared__ uint64_t smem[2048];

    for (int i = 0; i < num_vecs; i++) {
        hh::vec<uint64_t, 2> x;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            x[j] = input[i * 2048 + threadIdx.x * 2 + j];
        }
        kc::heap_parallel_insert<5>(x, i+1, heap, smem);
    }
}

__global__ void copy_from_heap(uint64_t *output, uint4* heap, int num_vecs) {

    __shared__ uint64_t smem[2048];

    for (int i = 0; i < num_vecs; i++) {
        hh::vec<uint64_t, 2> x = kc::load_heap_element<5>(1, heap);
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            output[i * 2048 + threadIdx.x * 2 + j] = x[j];
        }
        kc::heap_parallel_delete<5>(num_vecs-i, heap, smem);
    }
}

TEST(Heap, Sort) {

    int n = 10000;

    hh::PRNG pcg(1, 2, 3);

    uint64_t* device_buffer;
    uint4* heap;
    uint64_t* host_input;
    uint64_t* host_output;

    cudaMallocHost((void**) &host_input, n << 14);
    cudaMallocHost((void**) &host_output, n << 14);
    cudaMalloc((void**) &device_buffer, n << 14);
    cudaMalloc((void**) &heap, n << 14);

    for (int i = 0; i < 2048*n; i++) {
        host_input[i] = pcg.generate64();
    }

    cudaMemcpy(device_buffer, host_input, n << 14, cudaMemcpyHostToDevice);
    copy_to_heap<<<1, 1024>>>(device_buffer, heap, n);
    copy_from_heap<<<1, 1024>>>(device_buffer, heap, n);
    cudaMemcpy(host_output, device_buffer, n << 14, cudaMemcpyDeviceToHost);

    cudaFree(device_buffer);
    cudaFree(heap);

    std::sort(host_input, host_input + 2048*n);

    for (int i = 0; i < 2048*n; i++) {
        EXPECT_EQ(host_output[i], host_input[2048*n - i - 1]);
    }

    cudaFreeHost(host_input);
    cudaFreeHost(host_output);
}
