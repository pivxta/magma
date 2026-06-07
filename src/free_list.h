#pragma once
#include <type_traits>
#include <optional>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cassert>
 
enum class FreeListPolicy: uint8_t {
    FirstFit,
    BestFit,
};
 
template<typename T>
struct FreeListAllocation {
    T offset;
    T size;
};
 
template<typename T>
class FreeList {
public:
    static_assert(std::is_unsigned_v<T>, "Free list requires an unsigned integer type");
 
    FreeList() {
        this->capacity = 0;
        this->policy = FreeListPolicy::FirstFit;
    }

    FreeList(T capacity, FreeListPolicy policy) {
        this->capacity = capacity;
        this->policy = policy;
        ranges.push_back(FreeRange{0, capacity});
    }
 
    std::optional<FreeListAllocation<T>> allocate(T size, T alignment) {
        if (size == 0 || !is_power_of_two(alignment)) {
            return std::nullopt;
        }
 
        size_t best = ranges.size();
        T best_offset = 0;
        T best_metric = 0;
 
        for (size_t i = 0; i < ranges.size(); ++i) {
            const FreeRange& range = ranges[i];

            T aligned = align_up(range.offset, alignment);
            if (aligned < range.offset) {
                continue;
            }

            T padding = aligned - range.offset;
            if (padding > range.size || range.size - padding < size) {
                continue;
            }

            T metric = range.size - size - padding;
            if (best == ranges.size() 
                || (policy == FreeListPolicy::BestFit && metric < best_metric)) 
            {
                best = i;
                best_offset = aligned;
                best_metric = metric;
            }
 
            if (policy == FreeListPolicy::FirstFit) {
                break;
            }
        }
 
        if (best == ranges.size()) {
            return std::nullopt;
        }
 
        FreeRange range = ranges[best];
        T range_end = range.offset + range.size;
        T alloc_end = best_offset + size;
        T left = best_offset - range.offset;
        T right = range_end - alloc_end;
 
        if (left > 0 && right > 0) {
            ranges[best] = FreeRange{range.offset, left};
            ranges.insert(ranges.begin() + best + 1, FreeRange{alloc_end, right});
        } else if (left > 0) {
            ranges[best] = FreeRange{range.offset, left};
        } else if (right > 0) {
            ranges[best] = FreeRange{alloc_end, right};
        } else {
            ranges.erase(ranges.begin() + best);
        }
 
        return FreeListAllocation<T>{best_offset, size};
    }
 
    bool free(FreeListAllocation<T> allocation) {
        T offset = allocation.offset;
        T size = allocation.size;
 
        if (size == 0 || offset > capacity || size > capacity - offset) {
            return false;
        }
        T end = offset + size;
 
        size_t i = std::distance(ranges.begin(), std::lower_bound(
            ranges.begin(), 
            ranges.end(), 
            offset, 
            [](const FreeRange& range, T target_offset) {
                return range.offset < target_offset;
            }
        ));
 
        if (i > 0 && ranges[i - 1].offset + ranges[i - 1].size > offset) {
            return false;
        }
        if (i < ranges.size() && end > ranges[i].offset) {
            return false;
        }
 
        ranges.insert(ranges.begin() + i, FreeRange{offset, size});
 
        if (i + 1 < ranges.size() 
            && ranges[i].offset + ranges[i].size == ranges[i + 1].offset) 
        {
            ranges[i].size += ranges[i + 1].size;
            ranges.erase(ranges.begin() + i + 1);
        }
        if (i > 0 &&
            ranges[i - 1].offset + ranges[i - 1].size == ranges[i].offset) 
        {
            ranges[i - 1].size += ranges[i].size;
            ranges.erase(ranges.begin() + i);
        }
 
        return true;
    }
 
    void reset() {
        ranges.clear();
        ranges.push_back(FreeRange{0, capacity});
    }
 
private:
    static inline T align_up(T value, T alignment) {
        assert(is_power_of_two(alignment));
        return (value + alignment - 1) & ~(alignment - 1);
    }
 
    static inline bool is_power_of_two(T n) {
        return (n > 0) && ((n & (n - 1)) == 0);
    }
 
    struct FreeRange {
        T offset;
        T size;
    };
 
    std::vector<FreeRange> ranges;
    T capacity;
    FreeListPolicy policy;
};