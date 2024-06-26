
#include <silk/floyd.hpp>
#include <gtest/gtest.h>


__global__ void floyd_kernel(const uint32_t *input, int32_t *output) {

    uint32_t perturbation = input[blockIdx.x * 32 + threadIdx.x];
    uint32_t px = 0;
    uint32_t py = 0;

    uint32_t perturbed_time = 0;
    uint32_t restore_time = ~0;
    
    uint4 ad0; ad0.x = 0; ad0.y = 0; ad0.z = 0; ad0.w = 0;
    uint4 ad1; ad1.x = 0xffffffffu; ad1.y = 0xffffffffu; ad1.z = 0xffffffffu; ad1.w = 0xffffffffu;
    uint4 ad2 = ad1;
    uint4 al2 = ad1;
    uint4 al3 = ad1;
    uint4 ad4 = ad1;
    uint4 ad5 = ad1;
    uint4 ad6 = ad1;
    uint4 stator = ad0;
    uint4 exempt = ad0;

    int result = kc::floyd_cycle<false>(ad0, ad1, ad2, al2, al3, ad4, ad5, ad6, stator, exempt, perturbation, px, py, perturbed_time, restore_time, 28, 28, 784, 9999, 9999);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = result;
    }
}

TEST(Floyd, Cycle) {

    std::vector<std::vector<uint32_t>> problems = {
        {0x1d8, 0x385, 0xd09, 0xd09, 0x385, 0x1d8}, // copperhead
        {0x3ff}, // pentadecathlon precursor
        {0x02, 0xc0, 0x47}, // diehard
        {0x7f}, // honeyfarm precursor
        {0x1f}, // traffic light precursor
        {0x1f, 0x11}, // pulsar precursor
        {1, 7, 2}, // r-pentomino
    };

    std::vector<int32_t> solutions = {640, 15, 0, 1, 2, 3, -1};

    int N = problems.size();

    uint32_t* input_h;
    uint32_t* input_d;
    int32_t* output_d;
    int32_t* output_h;

    cudaMallocHost((void**) &input_h, N * 128);
    cudaMalloc((void**) &input_d, N * 128);
    cudaMalloc((void**) &output_d, N * 4);
    cudaMallocHost((void**) &output_h, N * 4);

    for (int i = 0; i < N * 32; i++) { input_h[i] = 0; }
    for (int i = 0; i < N; i++) {
        int K = problems[i].size();
        for (int j = 0; j < K; j++) {
            input_h[i * 32 + j + 16 - (K >> 1)] = problems[i][j] << 8;
        }
    }

    cudaMemcpy(input_d, input_h, N * 128, cudaMemcpyHostToDevice);
    floyd_kernel<<<N, 32>>>(input_d, output_d);
    cudaMemcpy(output_h, output_d, N * 4, cudaMemcpyDeviceToHost);

    cudaFree(input_d);
    cudaFree(output_d);

    for (int i = 0; i < N; i++) {
        EXPECT_EQ(output_h[i], solutions[i]);
    }

    cudaFreeHost(input_h);
    cudaFreeHost(output_h);
}
