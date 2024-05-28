/**
 * Program for testing the Silk NNUE by running it in evaluation
 * mode on a set of test vectors
 */

#include <silk/nnue.hpp>
#include <stdio.h>
#include <iostream>


__global__ void testnet_kernel(const float4* nnue, const uint32_t* samples, float* output) {

    uint32_t signature = samples[blockIdx.x * 32 + threadIdx.x] & 255;
    float loss = kc::evaluate_nnue(signature, nnue);
    if (threadIdx.x == 0) { output[blockIdx.x] = loss; }

}


int main(int argc, char* argv[]) {

    if (argc != 5) {
        std::cerr << "Usage: ./testnet nnue.dat samples.dat output.dat n_samples" << std::endl;
        return 1;
    }

    int n_samples = std::stoll(argv[4]);

    float4* nnue_d;
    uint32_t* samples_d;
    float* outputs_d;

    float4* nnue_h;
    uint32_t* samples_h;
    float* outputs_h;

    cudaMalloc((void**) &nnue_d, 3826176);
    cudaMalloc((void**) &samples_d, 128 * n_samples);
    cudaMalloc((void**) &outputs_d, 4 * n_samples);

    cudaMallocHost((void**) &nnue_h, 3826176);
    cudaMallocHost((void**) &samples_h, 128 * n_samples);
    cudaMallocHost((void**) &outputs_h, 4 * n_samples);

    {
        // load NNUE:
        FILE *fptr = fopen(argv[1], "r");
        fread(nnue_h, 512, 7473, fptr);
        fclose(fptr);
    }

    {
        // load samples:
        FILE *fptr = fopen(argv[2], "r");
        fread(samples_h, 128, n_samples, fptr);
        fclose(fptr);
    }

    cudaMemcpy(nnue_d, nnue_h, 3826176, cudaMemcpyHostToDevice);
    cudaMemcpy(samples_d, samples_h, 128 * n_samples, cudaMemcpyHostToDevice);
    testnet_kernel<<<n_samples, 32>>>(nnue_d, samples_d, outputs_d);
    cudaMemcpy(outputs_h, outputs_d, 4 * n_samples, cudaMemcpyDeviceToHost);

    {
        // save outputs:
        FILE* fptr = fopen(argv[3], "w");
        fwrite(outputs_h, 4, n_samples, fptr);
        fclose(fptr);
    }

    cudaFree(nnue_d);
    cudaFree(samples_d);
    cudaFree(outputs_d);

    cudaFreeHost(nnue_d);
    cudaFreeHost(samples_d);
    cudaFreeHost(outputs_d);

    return 0;

}
