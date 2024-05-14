#include <silk/torus.hpp>
#include <cpads/random/prng.hpp>
#include <gtest/gtest.h>

__global__ void active_bounds_kernel(uint32_t *ptr, int w, int h, int p) {

    uint32_t not_stable = ptr[threadIdx.x];
    uint32_t forced_stable = kc::get_forced_stable(not_stable, 0xffffffffu, w, h, p);
    ptr[threadIdx.x] = forced_stable;

}


inline void active_bounds_test(int ox, int oy, int ow, int oh, int w, int h, int p) {

    uint32_t* input;
    uint32_t* output;
    uint32_t* device_buffer;

    cudaMallocHost((void**) &input, 128);
    cudaMallocHost((void**) &output, 128);
    cudaMalloc((void**) &device_buffer, 128);

    // prepare input:
    for (int i = 0; i < 32; i++) { input[i] = 0; }
    input[oy] = (1u << ox);
    input[oy + oh - 1] = (1u << (ox + ow - 1));

    cudaMemcpy(device_buffer, input, 128, cudaMemcpyHostToDevice);
    active_bounds_kernel<<<1, 32>>>(device_buffer, w, h, p);
    cudaMemcpy(output, device_buffer, 128, cudaMemcpyDeviceToHost);

    cudaFree(device_buffer);

    bool good = true;

    for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 32; x++) {
            uint32_t actual = (output[y] >> x) & 1;
            uint32_t expected;
            if ((x < 2) || (x >= 30) || (y < 2) || (y >= 30)) {
                expected = 1;
            } else if (p <= 1) {
                expected = 1;
            } else if (p == 2) {
                if ((x == ox) && (y == oy)) {
                    expected = 0;
                } else if ((x == ox + ow - 1) && (y == oy + oh - 1)) {
                    expected = 0;
                } else {
                    expected = 1;
                }
            } else {
                if (x >= ox + w) {
                    expected = 1;
                } else if (x < ox + ow - w) {
                    expected = 1;
                } else if (y >= oy + h) {
                    expected = 1;
                } else if (y < oy + oh - h) {
                    expected = 1;
                } else {
                    expected = 0;
                }
            }
            if (actual != expected) { good = false; }
        }
    }

    if (!good) {
        std::cout << w << ", " << h << ", " << p << std::endl;
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                uint32_t actual = (output[y] >> x) & 1;
                uint32_t original = (input[y] >> x) & 1;
                std::cout << ".*@#"[actual + 2 * original];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        EXPECT_EQ(0, 1);
    }

    cudaFreeHost(input);
    cudaFreeHost(output);

}

inline void cartesian_check(int w, int h, int p) {
    for (int ox = 10; ox < 20; ox += 4) { // 3
        for (int oy = 10; oy < 20; oy += 7) { // 2
            for (int ow = 2; ow < 5; ow += 1) { // 3
                for (int oh = 2; oh < 8; oh += 2) { // 3
                    active_bounds_test(ox, oy, ow, oh, w, h, p);
                }
            }
        }
    }
}

TEST(ActiveBounds, 12x10x1) { cartesian_check(12, 10, 1); }
TEST(ActiveBounds, 12x10x2) { cartesian_check(12, 10, 2); }
TEST(ActiveBounds, 12x10x3) { cartesian_check(12, 10, 3); }
TEST(ActiveBounds, 7x5x3)   { cartesian_check(7, 5, 3); }
