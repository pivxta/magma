#include "time.h"
#include <chrono>

Instant Instant::now() {
    auto clock = std::chrono::steady_clock::now();
    auto nanoseconds = 
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock.time_since_epoch());
    return {static_cast<uint64_t>(nanoseconds.count())};
}