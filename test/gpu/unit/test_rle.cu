#include <silk/readrle.hpp>
#include <gtest/gtest.h>

TEST(RLE, Reader) {

    kc::ProblemHolder ph("examples/2c3.rle");

    auto solution = ph.complete();

    EXPECT_EQ(solution.size(), 64);

    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            std::cout << (((solution[y] >> x) & 1) ? '*' : '.');
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

}
