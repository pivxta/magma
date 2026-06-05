#pragma once
#include <optional>
#include <cassert>
#include <type_traits>

template<typename T>
struct ArenaAllocation {
    T offset;
    T size;
};

template<typename T>
class Arena {
public:
    static_assert(std::is_unsigned_v<T>, "Arena requires an unsigned integer type");

    Arena(T capacity) {
        this->capacity = capacity;
    }

    std::optional<ArenaAllocation<T>> allocate(T size, T alignment) {
        if (size == 0 || !is_power_of_two(alignment)) {
            return std::nullopt;
        }
        
        T offset = align_up(this->offset, alignment);
        if (size > this->capacity || offset > this->capacity - size) {
            return std::nullopt;
        }
        this->offset = offset + size;

        return ArenaAllocation<T>{
            .offset = offset,
            .size = size 
        };
    }

    void reset() {
        this->offset = 0; 
    }

private:
    static inline T align_up(T value, T alignment) {
        assert(is_power_of_two(alignment));
        return (value + alignment - 1) & ~(alignment - 1);
    }

    static inline bool is_power_of_two(T n) {
        return (n > 0) && ((n & (n - 1)) == 0);
    }

    T offset = 0;
    T capacity;
};