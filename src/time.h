#pragma once
#include <cstdint>
#include <type_traits>

class Instant {
public:
    Instant() = delete;

    [[nodiscard]]
    static Instant now();

    template<typename T>
    [[nodiscard]] T elapsed_seconds() const {
        if constexpr (std::is_floating_point_v<T>) {
            return static_cast<T>(elapsed_nanoseconds()) / static_cast<T>(1000000000.0);
        } else {
            return static_cast<T>(elapsed_nanoseconds() / 1000000000ULL);
        }
    }

    template<typename T>
    [[nodiscard]] T elapsed_milliseconds() const {
        if constexpr (std::is_floating_point_v<T>) {
            return static_cast<T>(elapsed_nanoseconds()) / static_cast<T>(1000000.0);
        } else {
            return static_cast<T>(elapsed_nanoseconds() / 1000000ULL);
        }
    }

    template<typename T>
    [[nodiscard]] T elapsed_microseconds() const {
        if constexpr (std::is_floating_point_v<T>) {
            return static_cast<T>(elapsed_nanoseconds()) / static_cast<T>(1000.0);
        } else {
            return static_cast<T>(elapsed_nanoseconds() / 1000ULL);
        }
    }

    [[nodiscard]]
    uint64_t elapsed_nanoseconds() const {
        return now().nanoseconds - this->nanoseconds;
    }

private:
    Instant(uint64_t nanoseconds) {
        this->nanoseconds = nanoseconds;
    }

    uint64_t nanoseconds;
};
