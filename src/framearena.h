#pragma once
#include "arena.h"
#include "buffer.h"
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>

template<typename T>
struct FrameSubBuffer {
public:
    vk::DeviceSize length() const {
        return this->elem_count;
    }

    vk::DeviceSize size_bytes() const {
        return this->local_range.size;
    }
    
    vk::DeviceAddress address() const {
        return this->base_address;
    }
    
    T* mapped() {
        return reinterpret_cast<T*>(this->mapped_data);
    }

    void write(const T& memory, vk::DeviceSize first = 0) {
        memcpy(this->mapped() + first, &memory, sizeof(T));
    }
    
    void write(std::span<T> memory, vk::DeviceSize first = 0) {
        memcpy(this->mapped() + first, memory.data(), memory.size_bytes());
    }

    bool flush(
        vma::Allocator allocator,
        vk::DeviceSize first = 0,
        std::optional<vk::DeviceSize> count = std::nullopt
    ) {
        vk::DeviceSize flush_size = this->local_range.size;
        if (count.has_value()) {
            flush_size = count.value() * sizeof(T);
        }

        return allocator.flushAllocation(
            this->parent_allocation, 
            this->base_offset + first * sizeof(T), 
            flush_size
        ) == vk::Result::eSuccess;
    }
private:
    friend class FrameArena;

    vma::Allocation parent_allocation;
    vk::DeviceAddress base_address;
    vk::DeviceSize base_offset;
    vk::DeviceSize elem_count;
    void* mapped_data;

    ArenaAllocation<vk::DeviceSize> local_range;
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
    std::optional<FrameSubBuffer<T>> add(
        const T& value, 
        vk::DeviceSize alignment = DEFAULT_ALIGNMENT
    ) {
        auto buffer = this->allocate<T>(1, alignment);
        if (!buffer.has_value()) {
            return std::nullopt;
        }
        buffer->write(value);
        buffer->flush(this->allocator);
        return buffer;
    }

    template<typename T>
    std::optional<FrameSubBuffer<T>> add_array(
        std::span<T> values, 
        vk::DeviceSize alignment = DEFAULT_ALIGNMENT
    ) {
        auto buffer = this->allocate<T>(values.size(), alignment);
        if (!buffer.has_value()) {
            return std::nullopt;
        }
        buffer->write(values);
        buffer->flush(this->allocator);
        return buffer;
    }

    template<typename T>
    std::optional<FrameSubBuffer<T>> allocate(
        vk::DeviceSize count = 1,
        vk::DeviceSize min_alignment = DEFAULT_ALIGNMENT
    ) {
        vk::DeviceSize alignment = std::max(alignof(T), min_alignment);
        auto alloc = this->arena.allocate(count * sizeof(T), alignment);
        if (!alloc.has_value()) {
            return std::nullopt;
        }

        vk::DeviceSize offset = this->stride * this->frame_index + alloc->offset;
        FrameSubBuffer<T> allocation;
        allocation.parent_allocation = this->buffer.allocation;
        allocation.base_address = this->buffer.address + offset;
        allocation.base_offset = offset;
        allocation.elem_count = count;
        allocation.mapped_data = this->buffer.mapped(offset);
        allocation.local_range = *alloc;
        return allocation;
    }

    void reset();

private:
    vma::Allocator allocator;

    uint32_t frame_index = 0;
    uint32_t frames_in_flight = 0;

    Buffer buffer;
    vk::DeviceSize stride = 0;

    Arena<vk::DeviceSize> arena;
};