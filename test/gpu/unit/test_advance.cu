#include <silk/unknown.hpp>
#include "still_lifes.hpp"

__global__ void advance_kernel(uint32_t *x, int gens) {

    uint32_t offset = blockIdx.x * 384 + threadIdx.x;

    uint32_t d0 = x[offset];
    uint32_t d1 = x[offset + 32];
    uint32_t d2 = x[offset + 64];
    uint32_t l2 = x[offset + 96];
    uint32_t l3 = x[offset + 128];
    uint32_t d4 = x[offset + 160];
    uint32_t d5 = x[offset + 192];
    uint32_t d6 = x[offset + 224];
    uint32_t not_low = x[offset + 256];
    uint32_t not_high = x[offset + 288];
    uint32_t not_stable = x[offset + 320];
    uint32_t stator = x[offset + 352];

    for (int i = 0; i < gens; i++) {
        kc::inplace_advance_unknown<false>(d0, d1, d2, l2, l3, d4, d5, d6, not_low, not_high, not_stable, stator);
    }

    x[offset] = d0;
    x[offset + 32] = d1;
    x[offset + 64] = d2;
    x[offset + 96] = l2;
    x[offset + 128] = l3;
    x[offset + 160] = d4;
    x[offset + 192] = d5;
    x[offset + 224] = d6;
    x[offset + 256] = not_low;
    x[offset + 288] = not_high;
    x[offset + 320] = not_stable;
}

std::vector<uint32_t> advance1gen(std::vector<uint32_t> a) {

    std::vector<uint32_t> b(32);

    for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 32; x++) {
            int weight = 0;
            for (int y2 = y - 1; y2 <= y + 1; y2++) {
                for (int x2 = x - 1; x2 <= x + 1; x2++) {
                    weight += (((x2 == x) && (y2 == y)) ? 1 : 2) * ((a[y2 & 31] >> (x2 & 31)) & 1);
                }
            }
            if ((weight >= 5) && (weight <= 7)) {
                b[y] |= (1u << x);
            }
        }
    }

    return b;
}

void check_fully_known(uint32_t *output) {

    int pop = 0;
    for (int i = 256; i < 352; i++) {
        pop += hh::popc32(output[i]);
    }

    EXPECT_EQ(pop, 2048);

    for (int i = 256; i < 288; i++) {
        EXPECT_EQ(output[i] ^ output[i + 32] ^ output[i + 64], ((uint32_t) 0));
    }
}

void check_unknown_stable(uint32_t *output) {

    check_fully_known(output);
    for (int i = 0; i < 32; i++) {
        EXPECT_EQ(output[i + 320], 0u);
    }
}

void check_advance(uint32_t *input, uint32_t *output, int gens) {

    for (int i = 0; i < 256; i++) {
        EXPECT_EQ(input[i], output[i]);
    }

    check_fully_known(output);

    std::vector<uint32_t> gen0(32);
    std::vector<uint32_t> genX(32);

    for (int i = 0; i < 32; i++) {
        uint32_t sl = ~(input[i + 96] & input[i + 128]);
        gen0[i] = sl ^ input[i + 320];
        genX[i] = sl ^ output[i + 320];
    }

    for (int i = 0; i < gens; i++) {
        gen0 = advance1gen(gen0);
    }

    for (int i = 0; i < 32; i++) {
        EXPECT_EQ(gen0[i], genX[i]);
    }
}

void generate_unknown_stable(uint64_t seed, uint32_t* output, double prob) {

    std::vector<uint32_t> gt(256);

    make_random_sl(seed, &(gt[0]), output, prob);

    for (int y = 0; y < 32; y++) {
        output[y + 256] = 0xffffffffu;
        output[y + 288] = 0xffffffffu;
        output[y + 320] = 0x00000000u;
        output[y + 352] = ((y >= 2) && (y < 30)) ? 0xc0000003u : 0xffffffffu;
    }
}

void random_sl_with_perturbation(uint64_t seed, uint32_t* output) {

    auto solution = make_random_sl(seed, output);

    std::vector<uint32_t> xordiff(32);

    hh::PRNG pcg(seed, 3511, 1093);

    for (int y = 12; y < 20; y++) {
        xordiff[y] = pcg.generate() & 0x000ff000u;
    }

    for (int y = 0; y < 32; y++) {
        output[y + 256] = (~solution[y]) | (~xordiff[y]);
        output[y + 288] = ( solution[y]) | (~xordiff[y]);
        output[y + 320] = xordiff[y];
        output[y + 352] = ((y >= 2) && (y < 30)) ? 0xc0000003u : 0xffffffffu;
    }
}

void advance_test(bool fully_stable) {

    int N = 100;
    int gens = 6;

    uint32_t* input;
    uint32_t* output;
    uint32_t* device_buffer;

    cudaMalloc((void**) &device_buffer, N * 1536);
    cudaMallocHost((void**) &input, N * 1536);
    cudaMallocHost((void**) &output, N * 1536);

    std::cout << "Generating problems: [";
    for (int i = 0; i < N; i++) {
        if (fully_stable) {
            generate_unknown_stable(i, input + i * 384, (1.0 * i) / N);
        } else {
            random_sl_with_perturbation(i, input + i * 384);
        }
        std::cout << "*" << std::flush;
    }
    std::cout << "]" << std::endl;

    cudaMemcpy(device_buffer, input, N * 1536, cudaMemcpyHostToDevice);
    advance_kernel<<<N, 32>>>(device_buffer, gens);
    cudaMemcpy(output, device_buffer, N * 1536, cudaMemcpyDeviceToHost);

    cudaFree(device_buffer);

    for (int i = 0; i < N; i++) {
        if (fully_stable) {
            check_unknown_stable(output + i * 384);
        } else {
            check_advance(input + i * 384, output + i * 384, gens);
        }
    }

    cudaFreeHost(input);
    cudaFreeHost(output);
}

TEST(Advance, Random) { advance_test(false); }
TEST(Advance, Stable) { advance_test(true); }
