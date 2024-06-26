#include <silk/stable.hpp>
#include "still_lifes.hpp"

template<bool branching>
__global__ void stableprop_kernel(uint32_t *x) {

    uint32_t offset = blockIdx.x * 256 + threadIdx.x;

    uint32_t d0 = x[offset];
    uint32_t d1 = x[offset + 32];
    uint32_t d2 = x[offset + 64];
    uint32_t l2 = x[offset + 96];
    uint32_t l3 = x[offset + 128];
    uint32_t d4 = x[offset + 160];
    uint32_t d5 = x[offset + 192];
    uint32_t d6 = x[offset + 224];

    if constexpr (branching) {
        __shared__ uint32_t smem[256];
        kc::branched_stableprop(smem, d0, d1, d2, l2, l3, d4, d5, d6);
    } else {
        kc::stableprop<false>(d0, d1, d2, l2, l3, d4, d5, d6);
    }

    x[offset] = d0;
    x[offset + 32] = d1;
    x[offset + 64] = d2;
    x[offset + 96] = l2;
    x[offset + 128] = l3;
    x[offset + 160] = d4;
    x[offset + 192] = d5;
    x[offset + 224] = d6;
}

void check_stableprop(uint32_t* ground_truth, uint32_t* problem, uint32_t* deduction, uint32_t* deduction2) {

    int gpop = 0;
    int opop = 0;
    int ppop = 0;
    int dpop = 0;
    int epop = 0;

    for (int i = 0; i < 256; i++) {
        opop += hh::popc32(problem[i]);
    }

    kc::weak_stableprop(problem);

    for (int i = 0; i < 256; i++) {
        uint32_t g = ground_truth[i];
        uint32_t p = problem[i];
        uint32_t d = deduction[i];
        uint32_t e = deduction2[i];

        // check ground_truth >= deduction2 >= deduction >= problem:
        EXPECT_EQ((g | e), g);
        EXPECT_EQ((e | d), e);
        EXPECT_EQ((p & d), p);

        gpop += hh::popc32(g);
        ppop += hh::popc32(p);
        dpop += hh::popc32(d);
        epop += hh::popc32(e);
    }

    std::cout << gpop << " >= " << epop << " >= " << dpop << " >= " << ppop << " >= " << opop << std::endl;

    if (dpop == gpop - 1) {
        // something is terribly wrong:
        uint32_t xors[32] = {0};
        for (int y = 0; y < 256; y++) {
            xors[y & 31] ^= deduction[y];
        }

        for (int y = 0; y < 256; y++) {
            for (int x = 0; x < 32; x++) {
                std::cout << "@_*."[((deduction[y] >> x) & 1) + ((xors[y & 31] >> x) & 1) * 2];
            }
            std::cout << std::endl;
            if ((y & 31) == 31) { std::cout << std::endl; }
        }
    }
}



TEST(CompleteStill, Minimise) {

    std::vector<uint64_t> known_live = {0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 3350888185856ull, 11448882298880ull, 27924104740864ull, 3350892380160ull, 13962052370432ull, 58082137997312ull, 3350892380160ull, 29041068867584ull, 27924104740864ull, 3350892380160ull, 49146424459264ull, 58082137997312ull, 3350892380160ull, 29041068867584ull, 27924104740864ull, 3350892380160ull, 49146424459264ull, 58082137735168ull, 3350892380160ull, 29041068867584ull, 10331918172160ull, 1151869124608ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull};

    std::vector<uint64_t> known_dead = {0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 5445196447744ull, 6143302696960ull, 7260266823680ull, 31833479446528ull, 21222319456256ull, 12286605918208ull, 67017851535360ull, 41327675047936ull, 42444639174656ull, 67017851535360ull, 21222319456256ull, 12286605918208ull, 67017851535360ull, 41327675047936ull, 42444639174656ull, 67017851535360ull, 21222319456256ull, 12286605918208ull, 67017851273216ull, 6143302696960ull, 7260266823680ull, 1047152033792ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull};

    auto constraints = kc::expand_constraints<uint64_t>(&(known_live[0]), &(known_dead[0]));

    auto solution = kc::complete_still_life<uint64_t>(&(constraints[0]), 4, true);

    int population = 0;

    for (int y = 0; y < 64; y++) {
        population += hh::popc64(solution[y]);
        for (int x = 0; x < 64; x++) {
            std::cout << (((solution[y] >> x) & 1) ? '*' : '.');
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "population = " << population << std::endl;
    EXPECT_EQ(population, 306);
}


TEST(CompleteStill, Random) {

    int N = 100;

    uint32_t* ground_truth;
    uint32_t* problem;
    uint32_t* deduction;
    uint32_t* deduction2;
    uint32_t* device_buffer;

    cudaMallocHost((void**) &ground_truth, N << 10);
    cudaMallocHost((void**) &problem, N << 10);
    cudaMallocHost((void**) &deduction, N << 10);
    cudaMallocHost((void**) &deduction2, N << 10);
    cudaMalloc((void**) &device_buffer, N << 10);

    std::cout << "Generating problems: [";
    for (int i = 0; i < N; i++) {
        auto solution = make_random_sl(i, ground_truth + (i << 8), problem + (i << 8), (1.00 * i) / N);
        std::cout << "*" << std::flush;
        /*
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                std::cout << (((solution[y] >> x) & 1) ? '*' : '.');
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        */
    }
    std::cout << "]" << std::endl;

    std::cout << "Running kernel..." << std::endl;
    cudaMemcpy(device_buffer, problem, N << 10, cudaMemcpyHostToDevice);
    stableprop_kernel<false><<<N, 32>>>(device_buffer);
    cudaMemcpy(deduction, device_buffer, N << 10, cudaMemcpyDeviceToHost);
    stableprop_kernel<true><<<N, 32>>>(device_buffer);
    cudaMemcpy(deduction2, device_buffer, N << 10, cudaMemcpyDeviceToHost);
    cudaFree(device_buffer);
    std::cout << "...done!" << std::endl;

    for (int i = 0; i < N; i++) {
        check_stableprop(ground_truth + (i << 8), problem + (i << 8), deduction + (i << 8), deduction2 + (i << 8));
    }

    cudaFreeHost(ground_truth);
    cudaFreeHost(problem);
    cudaFreeHost(deduction);
    cudaFreeHost(deduction2);
}
