#include <silk/hilbert.hpp>
#include <gtest/gtest.h>
#include <unordered_set>

TEST(Hilbert, 256x256) {

    int limit = 1;

    std::unordered_set<uint64_t> encountered;

    int32_t last_x = 0;
    int32_t last_y = 0;

    for (uint64_t p = 0; p < 65536; p++) {

        uint64_t z = kc::hilbert_d2xy(p);

        encountered.insert(z);

        int32_t y = ((int32_t) (z >> 32));
        int32_t x = ((int32_t) (z & 0xffffffffu));

        EXPECT_LT(x, limit);
        EXPECT_LT(y, limit);

        // check that curve is continuous and begins at (0, 0):
        int32_t dx = x - last_x; last_x = x;
        int32_t dy = y - last_y; last_y = y;
        EXPECT_EQ(dx * dx + dy * dy, ((p == 0) ? 0 : 1));

        if (p == limit * limit - 1) {
            // we've done a power of 4 elements:
            EXPECT_EQ(encountered.size(), limit * limit);
            limit *= 2;
        }
    }
}
