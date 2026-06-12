#pragma once
#include <optional>
#include <vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include "device.h"

class Buffer {
public:
    Buffer() = default;

    Buffer(const Buffer&) noexcept = default;
    Buffer& operator=(const Buffer&) noexcept = default;
    Buffer(Buffer&&) noexcept = default;
    Buffer& operator=(Buffer&&) noexcept = default;

    explicit Buffer(
        const DeviceHandle& device,
        vk::BufferCreateInfo buffer_info,
        const vma::AllocationCreateInfo& alloc_info
    );

    void destroy(const DeviceHandle& device) {
        if (this->buffer != vk::Buffer{}) {
            device->allocator.destroyBuffer(this->buffer, this->allocation);
        }
        this->buffer = vk::Buffer();
        this->allocation = vma::Allocation();
        this->size_ = 0;
        this->address_ = 0;
        this->mapped_data = nullptr;
    }

    vk::DeviceSize size() const {
        return this->size_;
    }

    vk::DeviceAddress address() const {
        return this->address_;
    }

    void* mapped(vk::DeviceSize offset_bytes = 0) {
        if (this->mapped_data == nullptr) {
            return nullptr;
        }
        
        return reinterpret_cast<uint8_t*>(this->mapped_data) + offset_bytes;
    }

    bool flush(
        const DeviceHandle& device,
        vk::DeviceSize offset = 0,
        std::optional<vk::DeviceSize> size = std::nullopt
    ) const {
        return device->allocator.flushAllocation(
            this->allocation, 
            offset, size.value_or(this->size_)
        ) == vk::Result::eSuccess;
    }

    operator vk::Buffer() const {
        return this->buffer;
    }

    operator vk::Buffer() {
        return this->buffer;
    }

private:
    vk::Buffer buffer;
    vma::Allocation allocation;
    vk::DeviceSize size_ = 0;
    vk::DeviceAddress address_ = 0;
    void *mapped_data = nullptr;
};