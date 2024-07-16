#include <cpads/random/prng.hpp>
#include <gtest/gtest.h>


/**
 * This kernel is incorrect on Clang but correct on nvcc.
 */
__global__ void buggy_kernel(const uint32_t *src, uint32_t *dst) {

    int i = blockIdx.x & 63;
    int ls = i & 63;
    int rs = (-i) & 63;
    uint32_t a = src[blockIdx.x * 32 + threadIdx.x];

    uint64_t x = a;
    x = (x << ls) | (x >> rs);
    a = ((uint32_t) x);

    dst[blockIdx.x * 32 + threadIdx.x] = a;
}


/**
 * This kernel is correct on both Clang and nvcc.
 */
__global__ void reference_kernel(const uint32_t *src, uint32_t *dst) {

    int i = blockIdx.x & 63;
    int ls = i & 63;
    int rs = (-i) & 63;
    uint32_t a = src[blockIdx.x * 32 + threadIdx.x];

    uint32_t x2 = a;
    a = 0;
    if (ls < 32) { a |= (x2 << ls); }
    if (rs < 32) { a |= (x2 >> rs); }

    dst[blockIdx.x * 32 + threadIdx.x] = a;
}


TEST(Clang, Bug) {

    constexpr int n = 128 * 64 * 64;
    uint32_t* d_a;
    uint32_t* d_b;
    uint32_t* d_c;
    uint32_t* h_a;
    uint32_t* h_b;
    uint32_t* h_c;

    cudaMalloc((void**) &d_a, n);
    cudaMalloc((void**) &d_b, n);
    cudaMalloc((void**) &d_c, n);
    cudaMallocHost((void**) &h_a, n);
    cudaMallocHost((void**) &h_b, n);
    cudaMallocHost((void**) &h_c, n);

    hh::PRNG pcg(1, 2, 3);

    for (int i = 0; i < (n >> 2); i++) {
        // generate some random data
        h_a[i] = pcg.generate();
    }

    cudaMemcpy(d_a, h_a, n, cudaMemcpyHostToDevice);
    reference_kernel<<<4096, 32>>>(d_a, d_b);
    buggy_kernel<<<4096, 32>>>(d_a, d_c);
    cudaMemcpy(h_b, d_b, n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < (n >> 2); i++) {
        EXPECT_EQ(h_b[i], h_c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
}
