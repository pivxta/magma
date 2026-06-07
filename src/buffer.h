#pragma once
#include <vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include "vk_error.h"

struct Buffer {
    vk::Buffer buffer;
    vma::Allocation allocation;
    vk::DeviceSize size;
    vk::DeviceAddress address;
    void *mapped_data;

    void destroy(vma::Allocator allocator) {
        allocator.destroyBuffer(this->buffer, this->allocation);
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

    void flush(
        vma::Allocator allocator,
        vk::DeviceSize offset = 0,
        std::optional<vk::DeviceSize> size = std::nullopt
    ) {
        vk_expect(allocator.flushAllocation(
            this->allocation, 
            offset, size.value_or(this->size)
        ), "Failed to flush buffer allocation");
    }

    operator vk::Buffer() const {
        return this->buffer;
    }

    operator vk::Buffer() {
        return this->buffer;
    }
};

Buffer create_buffer(
    vk::Device device,
    vma::Allocator allocator,
    vk::BufferCreateInfo buffer_info,
    const vma::AllocationCreateInfo& alloc_info
);

Buffer create_gpu_buffer(
    vk::Device device,
    vma::Allocator allocator,
    vk::BufferUsageFlags usage, 
    vk::DeviceSize size
);

Buffer create_mapped_buffer(
    vk::Device device,
    vma::Allocator allocator,
    vk::BufferUsageFlags usage, 
    vk::DeviceSize size
);

Buffer create_staging_buffer(
    vk::Device device,
    vma::Allocator allocator,
    vk::BufferUsageFlags usage, 
    vk::DeviceSize size
);