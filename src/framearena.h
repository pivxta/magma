#pragma once
#include "arena.h"
#include "buffer.h"
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>

struct FrameSubBuffer {
    vma::Allocation parent_allocation;
    vk::DeviceAddress base_address;
    vk::DeviceSize base_offset;
    void* mapped;

    ArenaAllocation<vk::DeviceSize> local_range;

    vk::DeviceSize size() const {
        return this->local_range.size;
    }
    
    template<typename T>
    T* mapped_as() {
        return reinterpret_cast<T*>(this->mapped);
    }

    template<typename T>
    void write(const T& memory, vk::DeviceSize offset = 0) {
        memcpy(this->mapped_as<uint8_t>() + offset, &memory, sizeof(T));
    }
    
    template<typename T>
    void write_array(std::span<T> memory, vk::DeviceSize offset = 0) {
        memcpy(this->mapped_as<uint8_t>() + offset, memory.data(), memory.size_bytes());
    }

    bool flush(
        vma::Allocator allocator,
        vk::DeviceSize offset = 0,
        std::optional<vk::DeviceSize> size = std::nullopt
    ) {
        return allocator.flushAllocation(
            this->parent_allocation, 
            this->base_offset + offset, 
            size.value_or(this->local_range.size)
        ) == vk::Result::eSuccess;
    }
};


class FrameArena {
public:
    static constexpr vk::DeviceSize DEFAULT_ALIGNMENT = 16;

    FrameArena() = default;
    FrameArena(
        vk::Device device, 
        vma::Allocator allocator, 
        vk::BufferUsageFlags usage,
        uint32_t frames_in_flight,
        vk::DeviceSize capacity_per_fif
    );
    void destroy();

    template<typename T>
    std::optional<FrameSubBuffer> add(
        const T& value, 
        vk::DeviceSize alignment = DEFAULT_ALIGNMENT
    ) {
        std::optional<FrameSubBuffer> buffer = this->allocate<T>(1, alignment);
        if (!buffer.has_value()) {
            return std::nullopt;
        }
        buffer->write(value);
        buffer->flush(this->allocator);
        return buffer;
    }

    template<typename T>
    std::optional<FrameSubBuffer> add_array(
        std::span<T> values, 
        vk::DeviceSize alignment = DEFAULT_ALIGNMENT
    ) {
        std::optional<FrameSubBuffer> buffer = this->allocate<T>(values.size(), alignment);
        if (!buffer.has_value()) {
            return std::nullopt;
        }
        buffer->write_array(values);
        buffer->flush(this->allocator);
        return buffer;
    }

    template<typename T>
    std::optional<FrameSubBuffer> allocate(
        vk::DeviceSize count = 1,
        vk::DeviceSize min_alignment = DEFAULT_ALIGNMENT
    ) {
        vk::DeviceSize alignment = std::max(alignof(T), min_alignment);
        return this->allocate(count * sizeof(T), alignment);
    }

    std::optional<FrameSubBuffer> allocate(
        vk::DeviceSize size,
        vk::DeviceSize alignment = DEFAULT_ALIGNMENT
    );

    void reset();

private:
    vma::Allocator allocator;

    uint32_t frame_index = 0;
    uint32_t frames_in_flight = 0;

    Buffer buffer;
    vk::DeviceSize stride = 0;

    Arena<vk::DeviceSize> arena;
};