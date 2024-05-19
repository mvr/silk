#include <silk/nnue.hpp>
#include <gtest/gtest.h>

TEST(NNUE, Disc) {

    int32_t ey[32] =  { 0, -3,  0,  3,
                        0, -2,  0,  2,
                        1, -2, -1,  2,
                        2, -2, -2,  2,
                        2, -1, -2,  1,
                        1, -1, -1,  1,
                        0, -1,  0,  1,
                        0,  0,  0,  0};

    int32_t ex[32] =  { 3,  0, -3,  0,
                        2,  0, -2,  0,
                        2,  1, -2, -1,
                        2,  2, -2, -2,
                        1,  2, -1, -2,
                        1,  1, -1, -1,
                        1,  0, -1,  0,
                        0,  0,  0,  0};

    for (int i = 0; i < 32; i++) {
        int32_t x = 0;
        int32_t y = 0;
        kc::lane2coords(i, x, y);
        EXPECT_EQ(x, ex[i]);       
        EXPECT_EQ(y, ey[i]);       
    }
}
