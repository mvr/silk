#include <cpads/core.hpp>
#include <gtest/gtest.h>

TEST(AAA, AAA) {

    size_t free_mem = 0;
    size_t total_mem = 0;

    EXPECT_EQ(hh::reportCudaError(cudaMemGetInfo(&free_mem, &total_mem)), 0);

    std::cerr << "Memory statistics: " << free_mem << " free; " << total_mem << " total." << std::endl;

}

