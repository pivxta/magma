#pragma once
#include <deque>
#include <type_traits>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include "buffer.h"
#include "free_list.h"

template<typename T>
requires std::is_trivially_copyable_v<T>
struct HeapSubBuffer {
public:
    vk::DeviceSize buffer_offset() const {
        return this->local_range.offset;
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
        return reinterpret_cast<T*>(this->mapped_data);
    }

    void write_mapped(const T& memory, vk::DeviceSize first = 0) {
        memcpy(this->mapped() + first, &memory, sizeof(T));
    }
    
    void write_mapped(std::span<const T> memory, vk::DeviceSize first = 0) {
        memcpy(this->mapped() + first, memory.data(), memory.size_bytes());
    }

    bool flush_mapped(
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
    friend class HeapBuffer;

    vma::Allocation parent_allocation;
    vk::DeviceAddress base_address;
    vk::DeviceSize base_offset;
    vk::DeviceSize elem_count;
    void* mapped_data;

    FreeListAllocation<vk::DeviceSize> local_range;
};

class HeapBuffer {
public:
    HeapBuffer() = default;
    HeapBuffer(
        vk::Device device, 
        vma::Allocator allocator,
        uint32_t frames_in_flight,
        vk::DeviceSize min_alignment,
        const vk::BufferCreateInfo& buffer_info,
        const vma::AllocationCreateInfo& alloc_info
    );
    void destroy();

    template<typename T>
    requires std::is_trivially_copyable_v<T>
    std::optional<HeapSubBuffer<T>> allocate(
        vk::DeviceSize count = 1,
        std::optional<vk::DeviceSize> min_alignment = std::nullopt
    ) {
        vk::DeviceSize alignment = 
            std::max({this->min_alignment, min_alignment.value_or(1), alignof(T)});

        auto alloc = this->free_list.allocate(count * sizeof(T), alignment);
        if (!alloc.has_value()) {
            return std::nullopt;
        }

        HeapSubBuffer<T> allocation;
        allocation.parent_allocation = this->buffer_.allocation;
        allocation.base_address = this->buffer_.address + alloc->offset;
        allocation.base_offset = alloc->offset;
        allocation.elem_count = count;
        allocation.mapped_data = this->buffer_.mapped(alloc->offset);
        allocation.local_range = *alloc;
        return allocation;
    }

    template<typename T>
    void deferred_free(const HeapSubBuffer<T>& allocation) {
        this->free_queue.push_front(PendingFree{
            .request_frame = this->frame_counter,
            .range = allocation.local_range
        });
    }

    template<typename T>
    bool immediate_free(const HeapSubBuffer<T>& allocation) {
        return this->free_list.free(allocation.local_range);
    }

    void begin_frame(uint64_t frame_counter);
    void free_pending();

    const Buffer& buffer() const {
        return this->buffer_;
    }

private:
    struct PendingFree {
        uint64_t request_frame;
        FreeListAllocation<vk::DeviceSize> range;
    };

    vma::Allocator allocator;

    Buffer buffer_;
    vk::DeviceSize min_alignment;
    uint32_t frames_in_flight;
    uint64_t frame_counter = 0;

    FreeList<vk::DeviceSize> free_list;
    std::deque<PendingFree> free_queue;
};