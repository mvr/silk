#pragma once

#include "../concurrentqueue/blockingconcurrentqueue.h"
#include "common.hpp"
#include <silk/readrle.hpp>
#include <silk/metrics.hpp>
#include <stdio.h>
#include <chrono>
#include <cstdarg>

#define MESSAGE_KILL_THREAD 0
#define MESSAGE_SOLUTION 1
#define MESSAGE_STATUS 2
#define MESSAGE_INIT 3

std::string format_string(const char* format, ...) {
    // Start with variadic arguments
    va_list args;
    va_start(args, format);

    // Determine the size needed to hold the formatted string
    int size = vsnprintf(nullptr, 0, format, args);
    va_end(args);

    // Create a buffer of the required size
    std::vector<char> buffer(size + 1); // +1 for null terminator

    // Print to the buffer
    va_start(args, format);
    vsnprintf(buffer.data(), buffer.size(), format, args);
    va_end(args);

    // Create a std::string from the buffer
    return std::string(buffer.data());
}

struct PrintMessage {
    uint64_t message_type;
    std::string contents;
};

struct SolutionMessage {
    uint64_t message_type;
    int64_t  return_code;
    uint32_t solution[1024];
    uint64_t perturbation[64];
    uint64_t metrics[64];
};

typedef moodycamel::BlockingConcurrentQueue<PrintMessage> PrintQueue;
typedef moodycamel::BlockingConcurrentQueue<SolutionMessage> SolutionQueue;

void status_thread_loop(int num_gpus, int num_cadical_threads, SolutionQueue* status_queue, SolutionQueue* solution_queue, PrintQueue* print_queue) {

    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = t0;
    auto t2 = t0;

    std::vector<uint64_t> totals(64);
    std::vector<uint64_t> individuals(64 * num_gpus);

    int finished_gpus = 0;
    uint64_t next_elapsed_secs = 1;
    uint64_t last_problems = totals[COUNTER_READING_HEAD];

    SolutionMessage item;
    while (finished_gpus < num_gpus) {
        status_queue->wait_dequeue(item);
        if (item.message_type == MESSAGE_KILL_THREAD) {
            finished_gpus += 1;
        } else if (item.message_type == MESSAGE_SOLUTION) {
            solution_queue->enqueue(item);
            continue;
        } else if (item.message_type == MESSAGE_STATUS) {
            int device_id = item.return_code;
            for (int i = 0; i < 64; i++) {
                totals[i] += item.metrics[i] - individuals[i + 64 * device_id];
                individuals[i + 64 * device_id] = item.metrics[i];
            }
        }

        t2 = std::chrono::high_resolution_clock::now();
        auto total_usecs = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();
        uint64_t elapsed_secs = total_usecs / 1000000;

        if ((elapsed_secs == 0) || (finished_gpus == num_gpus) || (elapsed_secs >= next_elapsed_secs)) {
            auto usecs = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            double total_mprobs_per_sec    = ((double) totals[COUNTER_READING_HEAD]) / ((double) total_usecs);
            double current_mprobs_per_sec  = ((double) (totals[COUNTER_READING_HEAD] - last_problems)) / ((double) usecs);

            t1 = t2;
            last_problems = totals[COUNTER_READING_HEAD];

            char time_specifier;
            uint64_t denom;
            uint64_t elapsed_increment;

            if (elapsed_secs >= 1209600) { // 336 hours --> 14 days
                time_specifier = 'd';
                denom = 86400;
                elapsed_increment = 86400;
            } else if (elapsed_secs >= 28800) { // 480 minutes --> 8 hours
                time_specifier = 'h';
                denom = 3600;
                elapsed_increment = (elapsed_secs >= 172800) ? 14400 : 3600;
            } else if (elapsed_secs >= 600) { // 600 seconds --> 10 minutes
                time_specifier = 'm';
                denom = 60;
                elapsed_increment = (elapsed_secs >= 7200) ? 900 : ((elapsed_secs >= 1800) ? 300 : 60);
            } else {
                time_specifier = 's';
                denom = 1;
                elapsed_increment = (elapsed_secs >= 60) ? 30 : ((elapsed_secs >= 10) ? 5 : 1);
            }

            PrintMessage pm; pm.message_type = MESSAGE_STATUS;

            if ((elapsed_secs == 0) || (finished_gpus == num_gpus)) {
                double elapsed_time = ((double) total_usecs) / (1.0e6 * ((double) denom));
                pm.contents += format_string("|%6.2f %c |", elapsed_time, time_specifier);
            } else {
                next_elapsed_secs += elapsed_increment;
                unsigned long long elapsed_time = elapsed_secs / denom;
                pm.contents += format_string("| %4d %c  |", elapsed_time, time_specifier);
            }

            uint64_t open_problems = totals[COUNTER_WRITING_HEAD] - totals[COUNTER_READING_HEAD];

            pm.contents += format_string("%14llu | %8llu (%5.2f%%) | %7llu | %7.3f | %7.3f | %7.3f | %7llu | %7llu |",
                ((unsigned long long) last_problems),
                ((unsigned long long) open_problems),
                (100.0 * open_problems / totals[METRIC_PRB_SIZE]),
                ((unsigned long long) totals[METRIC_BATCH_SIZE]),
                ((double) (totals[METRIC_ROLLOUT])) / last_problems,
                current_mprobs_per_sec,
                total_mprobs_per_sec,
                ((unsigned long long) (totals[COUNTER_SOLUTION_HEAD] - totals[METRIC_FIZZLE])),
                ((unsigned long long) totals[METRIC_FIZZLE])
            );

            print_queue->enqueue(pm);
        }
    }

    // tear down solution threads:
    for (int i = 0; i < num_cadical_threads; i++) {
        SolutionMessage sm;
        sm.message_type = MESSAGE_KILL_THREAD;
        solution_queue->enqueue(sm);
    }

    // alert print queue that we've finished:
    {
        PrintMessage pm;
        pm.message_type = MESSAGE_KILL_THREAD;
        print_queue->enqueue(pm);
    }
}

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

    // alert print queue that we've finished:
    {
        PrintMessage pm;
        pm.message_type = MESSAGE_KILL_THREAD;
        print_queue->enqueue(pm);
    }
}

void print_thread_loop(int num_writers, PrintQueue* print_queue) {

    int finished_writers = 0;

    uint64_t last_message_type = MESSAGE_INIT;

    PrintMessage item;
    while (finished_writers != num_writers) {
        print_queue->wait_dequeue(item);

        if (item.message_type == MESSAGE_KILL_THREAD) {
            finished_writers += 1;
            if (finished_writers != num_writers) { continue; }
        }

        if ((item.message_type == MESSAGE_STATUS) && (last_message_type != MESSAGE_STATUS)) {
            std::cout << "+---------+-----------------------------------+---------+---------+-------------------+-------------------+" << std::endl;
            std::cout << "| elapsed |           problems                | current | rollout | speed (Mprob/sec) |     solutions     |" << std::endl;
            std::cout << "|  clock  +---------------+-------------------+  batch  |   per   +---------+---------+---------+---------+" << std::endl;
            std::cout << "|   time  |     solved    |  open (pct full)  |   size  | problem | current | overall | oscill. | fizzles |" << std::endl;
            std::cout << "+---------+---------------+-------------------+---------+---------+---------+---------+---------+---------+" << std::endl;
        } else if ((item.message_type != MESSAGE_STATUS) && (last_message_type == MESSAGE_STATUS)) {
            std::cout << "+---------+---------------+-------------------+---------+---------+---------+---------+---------+---------+" << std::endl;
            std::cout << std::endl;
        }

        last_message_type = item.message_type;
        std::cout << item.contents << std::endl;
    }
}