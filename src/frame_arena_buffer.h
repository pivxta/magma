#pragma once
#include "arena.h"
#include "buffer.h"
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>

template<typename T>
requires std::is_trivially_copyable_v<T>
struct FrameSubBuffer {
public:
    const Buffer& buffer() const {
        assert(this->parent_buffer != nullptr);
        return *this->parent_buffer;
    }

    vk::DeviceSize buffer_offset() const {
        return this->base_offset;
    }

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
        assert(this->mapped_data != nullptr);
        return reinterpret_cast<T*>(this->mapped_data);
    }

    void write(const T& memory, vk::DeviceSize first = 0) {
        assert(this->mapped_data != nullptr);
        assert(first < this->elem_count);

        memcpy(this->mapped() + first, &memory, sizeof(T));
    }
    
    void write(std::span<const T> memory, vk::DeviceSize first = 0) {
        assert(this->mapped_data != nullptr);
        assert(first <= this->elem_count);
        assert(memory.size() <= this->elem_count - first);

        memcpy(this->mapped() + first, memory.data(), memory.size_bytes());
    }

    bool flush(
        const DeviceHandle& device,
        vk::DeviceSize first = 0,
        std::optional<vk::DeviceSize> count = std::nullopt
    ) {
        assert(this->parent_buffer != nullptr);
        assert(first <= this->elem_count);

        vk::DeviceSize flush_count = count.value_or(this->elem_count - first);
        assert(flush_count <= this->elem_count - first);

        vk::DeviceSize flush_offset = first * sizeof(T);
        vk::DeviceSize flush_size = flush_count * sizeof(T);

        return this->parent_buffer->flush(
            device,
            this->base_offset + flush_offset,
            flush_size
        );
    }

private:
    friend class FrameArenaBuffer;

    const Buffer* parent_buffer = nullptr;
    vk::DeviceAddress base_address = 0;
    vk::DeviceSize base_offset = 0;
    vk::DeviceSize elem_count = 0;
    void* mapped_data = nullptr;

    ArenaAllocation<vk::DeviceSize> local_range;
};

class FrameArenaBuffer {
public:
    static constexpr vk::DeviceSize DEFAULT_ALIGNMENT = 16;

    FrameArenaBuffer() = default;
    FrameArenaBuffer(
        DeviceHandle device,
        vk::BufferUsageFlags usage,
        uint32_t frames_in_flight,
        vk::DeviceSize capacity_per_fif
    );

    FrameArenaBuffer(const FrameArenaBuffer&) = delete;
    FrameArenaBuffer& operator=(const FrameArenaBuffer&) = delete;
    FrameArenaBuffer(FrameArenaBuffer&&) noexcept = default;
    FrameArenaBuffer& operator=(FrameArenaBuffer&&) noexcept = default;

    ~FrameArenaBuffer();

    template<typename T>
    requires std::is_trivially_copyable_v<T>
    std::optional<FrameSubBuffer<T>> add(
        const T& value, 
        vk::DeviceSize alignment = DEFAULT_ALIGNMENT
    ) {
        auto buffer = this->allocate<T>(1, alignment);
        if (!buffer.has_value()) {
            return std::nullopt;
        }
        buffer->write(value);
        buffer->flush(this->device);
        return buffer;
    }

    template<typename T>
    requires std::is_trivially_copyable_v<T>
    std::optional<FrameSubBuffer<T>> add(
        std::span<const T> values, 
        vk::DeviceSize alignment = DEFAULT_ALIGNMENT
    ) {
        if (values.empty()) {
            return FrameSubBuffer<T>{};
        }

        auto buffer = this->allocate<T>(values.size(), alignment);
        if (!buffer.has_value()) {
            return std::nullopt;
        }
        buffer->write(values);
        buffer->flush(this->device);
        return buffer;
    }

    template<typename T>
    requires std::is_trivially_copyable_v<T>
    std::optional<FrameSubBuffer<T>> add(
        const std::vector<T>& values, 
        vk::DeviceSize alignment = DEFAULT_ALIGNMENT
    ) {
        return this->add<T>(std::span(values), alignment);
    }

    template<typename T>
    requires std::is_trivially_copyable_v<T>
    std::optional<FrameSubBuffer<T>> allocate(
        vk::DeviceSize count = 1,
        vk::DeviceSize min_alignment = DEFAULT_ALIGNMENT
    ) {
        if (count == 0) {
            return FrameSubBuffer<T>{};
        }

        vk::DeviceSize alignment = std::max(alignof(T), min_alignment);
        auto alloc = this->arena.allocate(count * sizeof(T), alignment);
        if (!alloc.has_value()) {
            return std::nullopt;
        }

        vk::DeviceSize offset = this->stride * this->frame_index + alloc->offset;
        FrameSubBuffer<T> allocation;
        allocation.parent_buffer = &this->buffer;
        allocation.base_address = this->buffer.address + offset;
        allocation.base_offset = offset;
        allocation.elem_count = count;
        allocation.mapped_data = this->buffer.mapped(offset);
        allocation.local_range = *alloc;
        return allocation;
    }

    void begin_frame(uint32_t next_frame_index);

private:
    DeviceHandle device;

    uint32_t frame_index = 0;

    Buffer buffer;
    vk::DeviceSize stride = 0;

    Arena<vk::DeviceSize> arena;
};