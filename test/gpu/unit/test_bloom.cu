#include <silk/bloom.hpp>
#include <cpads/random/prng.hpp>
#include <gtest/gtest.h>

__global__ void bloom_test_kernel(uint32_t *res) {

    int k = (threadIdx.x >> 5) + 1;

    uint32_t x = res[blockIdx.x * blockDim.x + threadIdx.x];
    x = kc::sparse_random(x, k);
    res[blockIdx.x * blockDim.x + threadIdx.x] = x;

}

TEST(Bloom, SparseRandom) {

    int N = 100000;

    uint32_t* a_h;
    uint32_t* a_d;

    cudaMallocHost((void**) &a_h, 128 * N);
    cudaMalloc((void**) &a_d, 128 * N);

    hh::PRNG pcg(1, 2, 3);

    for (int i = 0; i < 32*N; i++) {
        a_h[i] = pcg.generate();
    }

    cudaMemcpy(a_d, a_h, 128 * N, cudaMemcpyHostToDevice);
    bloom_test_kernel<<<(N >> 5), 1024>>>(a_d);
    cudaMemcpy(a_h, a_d, 128 * N, cudaMemcpyDeviceToHost);

    cudaFree(a_d);

    for (int i = 0; i < N; i++) {
        int p = 0;
        for (int j = 0; j < 32; j++) {
            p += hh::popc32(a_h[i*32+j]);
        }
        EXPECT_EQ(p, (i & 31) + 1);
    }

    cudaFreeHost(a_h);

}
