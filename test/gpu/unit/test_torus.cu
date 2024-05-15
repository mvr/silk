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


__global__ void plane_shift_kernel(const uint4 *src, uint4 *dst) {

    uint4 a = src[blockIdx.x * 32 + threadIdx.x];
    uint4 b;

    b.x = kc::shift_plane<false,  1>(a.x);
    b.y = kc::shift_plane<true,   1>(a.y);
    b.z = kc::shift_plane<false, -1>(a.z);
    b.w = kc::shift_plane<true,  -1>(a.w);

    dst[blockIdx.x * 32 + threadIdx.x] = b;
}


__global__ void plane_next_kernel(const uint32_t *src, uint32_t *dst) {

    uint32_t p = blockIdx.x;
    uint32_t mask = src[blockIdx.x * 32 + threadIdx.x];
    uint32_t q = kc::compute_next_cell(mask, p);

    dst[blockIdx.x] = q;

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


uint32_t plane_next_ref(const uint32_t* mask, uint32_t p) {

    int pop = 0;
    for (int i = 0; i < 32; i++) { pop += hh::popc32(mask[i]); }
    if (pop == 0) { return 0xffffffffu; }

    uint32_t q = p + 1;
    while (((mask[(q >> 5) & 31] >> (q & 31)) & 1) == 0) { q += 1; }
    return q;
}


void plane_next_test(int log2sparsity) {

    int n = 10000;

    uint32_t* d_a;
    uint32_t* d_b;
    uint32_t* h_a;
    uint32_t* h_b;

    cudaMalloc((void**) &d_a, n * 128);
    cudaMalloc((void**) &d_b, n * 4);
    cudaMallocHost((void**) &h_a, n * 128);
    cudaMallocHost((void**) &h_b, n * 4);

    hh::PRNG pcg(1, 2, 3);

    for (int i = 0; i < n * 32; i++) {
        uint32_t x = 0xffffffffu;
        for (int j = 0; j < log2sparsity; j++) {
            x &= pcg.generate();
        }
        h_a[i] = x;
    }

    cudaMemcpy(d_a, h_a, 128 * n, cudaMemcpyHostToDevice);
    plane_next_kernel<<<n, 32>>>(d_a, d_b);
    cudaMemcpy(h_b, d_b, 4 * n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        uint32_t actual = h_b[i];
        uint32_t expected = plane_next_ref(h_a + 32 * i, i);
        EXPECT_EQ(actual, expected);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
}


TEST(Next, s8) { plane_next_test(3); }
TEST(Next, s64) { plane_next_test(6); }
TEST(Next, s512) { plane_next_test(9); }


TEST(Plane, GetMiddle) {

    EXPECT_EQ(kc::get_middle(0x00ffff00u), ((uint32_t) 0));
    EXPECT_EQ(kc::get_middle(0x0ffff000u), ((uint32_t) 4));

}


TEST(Plane, Shift) {

    constexpr int n = 4096;
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
    plane_shift_kernel<<<(n / 512), 32>>>(d_a, d_b);
    cudaMemcpy(h_b, d_b, n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < (n >> 2); i += 128) {
        for (int j = 0; j < 32; j++) {
            EXPECT_EQ(h_b[i + 4*j + 0], h_a[i + 4*j + 0] << 1);
            EXPECT_EQ(h_b[i + 4*j + 2], h_a[i + 4*j + 2] >> 1);
            uint32_t c = 0;
            if (j >= 1) { c = h_a[i + 4*j + 1 - 4]; }
            EXPECT_EQ(h_b[i + 4*j + 1], c);
            uint32_t d = 0;
            if (j < 31) { d = h_a[i + 4*j + 3 + 4]; }
            EXPECT_EQ(h_b[i + 4*j + 3], d);
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
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
