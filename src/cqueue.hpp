#pragma once

#include "../concurrentqueue/blockingconcurrentqueue.h"
#include "common.hpp"
#include <silk/readrle.hpp>

#define MESSAGE_KILL_THREAD 0
#define MESSAGE_SOLUTION_STRING 1
#define MESSAGE_STATUS_STRING 2
#define MESSAGE_INIT_STRING 3

struct PrintMessage {
    uint64_t message_type;
    std::string contents;
};

struct SolutionMessage {
    uint64_t message_type;
    int64_t  return_code;
    uint32_t solution[1024];
    uint64_t perturbation[64];
};

typedef moodycamel::BlockingConcurrentQueue<PrintMessage> PrintQueue;
typedef moodycamel::BlockingConcurrentQueue<SolutionMessage> SolutionQueue;

void solution_thread_loop(SolutionQueue* solution_queue, PrintQueue* print_queue) {

    SolutionMessage item;
    while (true) {
        solution_queue->wait_dequeue(item);
        if (item.message_type == MESSAGE_KILL_THREAD) { break; }

        // unswizzle
        uint64_t tmp[512];
        for (int z = 0; z < 8; z++) {
            for (int y = 0; y < 32; y++) {
                tmp[64 * z + y]      = item.solution[128 * z + 4 * y    ] | (((uint64_t) item.solution[128 * z + 4 * y + 1]) << 32);
                tmp[64 * z + y + 32] = item.solution[128 * z + 4 * y + 2] | (((uint64_t) item.solution[128 * z + 4 * y + 3]) << 32);
            }
        }

        // attempt to stabilise:
        std::vector<uint64_t> res;
        for (int r = 4; r < 9; r++) {
            res = kc::complete_still_life(tmp, r, true);
            if (res.size() > 0) { break; }
        }

        if (res.size() > 0) {
            std::ostringstream ss;

            // include description of solution type:
            if (item.return_code == 0) {
                ss << "# Found fizzle" << std::endl;
            } else if (item.return_code == 1) {
                ss << "# Found restabilisation" << std::endl;
            } else {
                ss << "# Found oscillator with period " << item.return_code << std::endl;
            }

            // prune:
            int miny = 0;
            int maxy = 63;
            while ((item.perturbation[miny] == 0) && (res[miny] == 0)) { miny++; }
            while ((item.perturbation[maxy] == 0) && (res[maxy] == 0)) { maxy--; }

            // stringify:
            for (int y = miny; y <= maxy; y++) {
                for (int x = 0; x < 64; x++) {
                    ss << (((item.perturbation[y] >> x) & 1) ? 'o' : (((res[y] >> x) & 1) ? '*' : '.'));
                }
                ss << std::endl;
            }

            PrintMessage pm;
            pm.contents = ss.str();
            pm.message_type = item.message_type;
            print_queue->enqueue(pm);
        }
    }
}

void print_thread_loop(PrintQueue* print_queue) {

    uint64_t last_message_type = MESSAGE_INIT_STRING;

    PrintMessage item;
    while (true) {
        print_queue->wait_dequeue(item);

        if ((item.message_type == MESSAGE_STATUS_STRING) && (last_message_type != MESSAGE_STATUS_STRING)) {
            std::cout << "+---------+-----------------------------------+---------+---------+-------------------+-------------------+" << std::endl;
            std::cout << "| elapsed |           problems                | current | rollout | speed (Mprob/sec) |     solutions     |" << std::endl;
            std::cout << "|  clock  +---------------+-------------------+  batch  |   per   +---------+---------+---------+---------+" << std::endl;
            std::cout << "|   time  |     solved    |  open (pct full)  |   size  | problem | current | overall | oscill. | fizzles |" << std::endl;
            std::cout << "+---------+---------------+-------------------+---------+---------+---------+---------+---------+---------+" << std::endl;
        } else if ((item.message_type != MESSAGE_STATUS_STRING) && (last_message_type == MESSAGE_STATUS_STRING)) {
            std::cout << "+---------+---------------+-------------------+---------+---------+---------+---------+---------+---------+" << std::endl;
            std::cout << std::endl;
        }

        last_message_type = item.message_type;
        if (item.message_type == MESSAGE_KILL_THREAD) { break; }
        std::cout << item.contents << std::endl;
    }
}
