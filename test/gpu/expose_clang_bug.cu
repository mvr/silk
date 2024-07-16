/**
 * This source file contains a self-contained test case, namely a
 * kernel that compiles incorrectly with Clang but correctly on nvcc.
 *
 * The bug is present on at least the following versions of Clang:
 *  -- 17.0.6
 * when compiling for at least the following GPU compute capabilities:
 *  -- sm_61
 *
 * Details are included at the bottom of the file.
 */


// We use the fixed-width datatypes uint32_t and uint64_t.
#include <stdint.h>


/**
 * These includes are only used in host-side code; the GPU kernels
 * themselves have no header dependencies other than <stdint.h>
 */
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

void run_clang_bug_test(int blocks) {

    int n = 128 * blocks;
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
    reference_kernel<<<blocks, 32>>>(d_a, d_b);
    buggy_kernel<<<blocks, 32>>>(d_a, d_c);
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

/**
 * With 128 blocks we expose the bug because we have shift amounts
 * outside the range [0, 63]. Even though our kernel explicitly
 * has an "& 63" to reduce the range, this gets optimised out.
 *
 * If you inspect the generated SASS from each of nvcc and Clang,
 * the only difference (up to register renaming and reordering
 * instructions) is a "LOP32I.AND R0, R4, 0x3f;" that is present
 * in the nvcc output but absent in the Clang output.
 *
 * My theory is that it is optimised out by virtue of being the
 * input to a funnel shift, which has wraparound semantics, so
 * the "& 63" is not necessary. But then at some point later one
 * of the funnel shifts gets optimised into an ordinary shift --
 * an optimisation that is only valid if the shift amount has
 * been mapped into the range [0, 63].
 */
TEST(Clang, Bug) { run_clang_bug_test(128); }

/**
 * With 64 blocks the bug is not exposed.
 */
TEST(Clang, Nonbug) { run_clang_bug_test(64); }
