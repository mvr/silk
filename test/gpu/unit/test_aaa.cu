#include <cpads/core.hpp>
#include <gtest/gtest.h>

/**
 * This test runs alphabetically before all other tests so that it
 * can check that the CUDA runtime is working correctly before the
 * other unit tests run.
 */
TEST(Aaa, Aaa) {

    size_t free_mem = 0;
    size_t total_mem = 0;

    EXPECT_EQ(hh::reportCudaError(cudaMemGetInfo(&free_mem, &total_mem)), 0);

    std::cerr << "Memory statistics: " << free_mem << " free; " << total_mem << " total." << std::endl;

}

