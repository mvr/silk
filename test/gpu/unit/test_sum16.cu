#include <silk/unknown.hpp>
#include <cpads/random/prng.hpp>
#include <gtest/gtest.h>

template<bool SubtractOne>
__global__ void sum16_kernel(uint4 *x) {

    uint4 a = x[blockIdx.x * 32 + threadIdx.x];
    uint4 b = kc::sum16<SubtractOne>(a.x, a.y);
    x[blockIdx.x * 32 + threadIdx.x] = b;

}

void check_sum16(uint32_t* input, uint32_t* output, bool SubtractOne) {

    for (int y = 1; y < 31; y++) {
        for (int x = 1; x < 31; x++) {

            int expected = 0;
            int actual = 0;

            for (int oy = y - 1; oy <= y + 1; oy++) {
                for (int ox = x - 1; ox <= x + 1; ox++) {
                    if ((ox == x) && (oy == y)) { continue; }
                    expected += ((input[oy * 4] >> ox) & 1);
                    expected += ((input[oy * 4 + 1] >> ox) & 1);
                }
            }

            // 4-bit saturating counter
            expected = hh::min(hh::max(0, expected - SubtractOne), 15);

            actual +=  (output[y * 4    ] >> x) & 1;
            actual += ((output[y * 4 + 1] >> x) & 1) * 2;
            actual += ((output[y * 4 + 2] >> x) & 1) * 4;
            actual += ((output[y * 4 + 3] >> x) & 1) * 8;

            EXPECT_EQ(actual, expected);
        }
    }
}

template<bool SubtractOne>
void run_sum16_test() {

    int N = 1000;

    uint32_t* input;
    uint32_t* output;
    uint4* device_buffer;

    cudaMallocHost((void**) &input, N << 9);
    cudaMallocHost((void**) &output, N << 9);
    cudaMalloc((void**) &device_buffer, N << 9);

    hh::PRNG pcg(1, 2, 3);

    for (int i = 0; i < N * 128; i++) {
        input[i] = pcg.generate();
    }

    cudaMemcpy(device_buffer, input, N << 9, cudaMemcpyHostToDevice);
    sum16_kernel<SubtractOne><<<N, 32>>>(device_buffer);
    cudaMemcpy(output, device_buffer, N << 9, cudaMemcpyDeviceToHost);

    cudaFree(device_buffer);

    for (int i = 0; i < N; i++) {
        check_sum16(input + i * 128, output + i * 128, SubtractOne);
    }

    cudaFreeHost(input);
    cudaFreeHost(output);

}

TEST(Sum16, Ordinary) { run_sum16_test<false>(); }
TEST(Sum16, Subtracted) { run_sum16_test<true>(); }
