#pragma once
#include <vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include "device.h"

struct Buffer {
    vk::Buffer buffer;
    vma::Allocation allocation;
    vk::DeviceSize size = 0;
    vk::DeviceAddress address = 0;
    void *mapped_data = nullptr;

    void destroy(const DeviceHandle& device) {
        device->allocator.destroyBuffer(this->buffer, this->allocation);
        this->buffer = vk::Buffer();
        this->allocation = vma::Allocation();
        this->size = 0;
        this->address = 0;
        this->mapped_data = nullptr;
    }

    void* mapped(vk::DeviceSize offset_bytes) {
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
            offset, size.value_or(this->size)
        ) == vk::Result::eSuccess;
    }

    operator vk::Buffer() const {
        return this->buffer;
    }

    operator vk::Buffer() {
        return this->buffer;
    }
};

Buffer create_buffer(
    const DeviceHandle& device,
    vk::BufferCreateInfo buffer_info,
    const vma::AllocationCreateInfo& alloc_info
);

Buffer create_gpu_buffer(
    const DeviceHandle& device,
    vk::BufferUsageFlags usage, 
    vk::DeviceSize size
);

Buffer create_mapped_buffer(
    const DeviceHandle& device,
    vk::BufferUsageFlags usage, 
    vk::DeviceSize size
);

Buffer create_staging_buffer(
    const DeviceHandle& device,
    vk::BufferUsageFlags usage, 
    vk::DeviceSize size
);