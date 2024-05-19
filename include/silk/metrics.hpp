#pragma once
#include <cpads/core.hpp>

#define METRIC_DOMINO_RULE 1
#define METRIC_STABLEPROP  2
#define METRIC_FLOYD_ITER  3
#define METRIC_ADVANCE     4
#define METRIC_ROLLOUT     5
#define METRIC_BRANCHING   6
#define METRIC_KERNEL      7

namespace kc {

_DI_ void bump_counter(uint32_t* metrics, int id) {
    if (metrics != nullptr) {
        if (threadIdx.x == 0) {
            atomicAdd(metrics, ((uint32_t) 1));
        }
    }
}

}
