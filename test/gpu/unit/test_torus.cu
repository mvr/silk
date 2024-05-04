#include <silk/torus.hpp>
#include <cpads/random/prng.hpp>
#include <gtest/gtest.h>


__global__ void torus_shift_kernel(const uint4 *src, uint4 *dst) {

    int i = blockIdx.x & 63;
    int j = blockIdx.x >> 6;

    uint4 a = src[blockIdx.x * 32 + threadIdx.x];
    uint4 b = kc::shift_torus(a, i, j);

    dst[blockIdx.x * 32 + threadIdx.x] = b;

}


void check_shift(uint32_t* h_a, uint32_t* h_b, int x, int y) {

    uint64_t a[64];
    uint64_t b[64];

    for (int i = 0; i < 32; i++) {
        a[i]      = h_a[4*i  ] + (((uint64_t) h_a[4*i+1]) << 32);
        a[i + 32] = h_a[4*i+2] + (((uint64_t) h_a[4*i+3]) << 32);
        b[i]      = h_b[4*i  ] + (((uint64_t) h_b[4*i+1]) << 32);
        b[i + 32] = h_b[4*i+2] + (((uint64_t) h_b[4*i+3]) << 32);
    }

    for (int i = 0; i < 64; i++) {
        uint64_t c = a[i];
        if (x != 0) { c = (c << x) | (c >> (64-x)); }
        EXPECT_EQ(c, b[(i + y) & 63]);
    }
}


TEST(Torus, Shift) {

    constexpr int n = 512 * 64 * 64;

    uint4* d_a;
    uint4* d_b;
    uint32_t* h_a;
    uint32_t* h_b;

    cudaMalloc((void**) &d_a, n);
    cudaMalloc((void**) &d_b, n);
    cudaMallocHost((void**) &h_a, n);
    cudaMallocHost((void**) &h_b, n);

    hh::PRNG pcg(1, 2, 3);

    for (int i = 0; i < (n >> 2); i++) {
        // generate some random data
        h_a[i] = pcg.generate();
    }

    cudaMemcpy(d_a, h_a, n, cudaMemcpyHostToDevice);
    torus_shift_kernel<<<4096, 32>>>(d_a, d_b);
    cudaMemcpy(h_b, d_b, n, cudaMemcpyDeviceToHost);

    for (int j = 0; j < 63; j++) {
        for (int i = 0; i < 63; i++) {
            int k = j * 64 + i;
            check_shift(h_a + 128 * k, h_b + 128 * k, i, j);
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);

}
