#include <silk/perm.hpp>
#include <gtest/gtest.h>
#include <algorithm>

void perm_test(uint64_t modulus, bool early_exit = false) {

    std::vector<uint64_t> x(modulus);

    for (uint64_t i = 0; i < modulus; i++) {
        x[i] = kc::random_perm(i, modulus, 42);
    }

    if (early_exit) { return; }

    std::sort(&(x[0]), &(x[modulus]));

    for (uint64_t i = 0; i < modulus; i++) {
        EXPECT_EQ(x[i], i);
    }
}

TEST(Perm,   65536) { perm_test(  65536); }
TEST(Perm,  262144) { perm_test( 262144); }
TEST(Perm, 1048576) { perm_test(1048576); }
TEST(Perm, 4194304) { perm_test(4194304); }

TEST(Perm, Perf) { perm_test(4194304, true); }
