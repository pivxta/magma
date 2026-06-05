#pragma once
#include <vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include "vkerror.h"

struct Buffer {
    vk::Buffer buffer;
    vma::Allocation allocation;
    vk::DeviceSize size;
    vk::DeviceAddress address;
    void *mapped;

    void destroy(vma::Allocator allocator) {
        allocator.destroyBuffer(this->buffer, this->allocation);
        this->buffer = vk::Buffer();
        this->allocation = vma::Allocation();
        this->size = 0;
        this->address = 0;
        this->mapped = nullptr;
    }

    void* get_mapped(vk::DeviceSize offset_bytes) {
        return reinterpret_cast<uint8_t*>(this->mapped) + offset_bytes;
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

static Buffer create_buffer(
    vk::Device device,
    vma::Allocator allocator,
    vk::BufferCreateInfo buffer_info,
    const vma::AllocationCreateInfo& alloc_info
) {
    vma::AllocationInfo info;
    auto [result, resources] = allocator.createBuffer(buffer_info, alloc_info, &info);
    vk_expect(result, "Failed to create buffer");

    vk::Buffer buffer = resources.second;
    vma::Allocation allocation = resources.first;
    vk::DeviceAddress address = 0;
    if (buffer_info.usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
        address = device.getBufferAddress(vk::BufferDeviceAddressInfo().setBuffer(buffer));
    }

    return {
        .buffer = buffer,
        .allocation = allocation,
        .size = buffer_info.size,
        .address = address,
        .mapped = info.pMappedData
    };
}

static inline Buffer create_gpu_buffer(
    vk::Device device,
    vma::Allocator allocator,
    vk::BufferUsageFlags usage, 
    vk::DeviceSize size
) {
    return create_buffer(
        device,
        allocator,
        vk::BufferCreateInfo()
            .setSharingMode(vk::SharingMode::eExclusive)
            .setSize(size)
            .setUsage(usage | vk::BufferUsageFlagBits::eShaderDeviceAddress),
        vma::AllocationCreateInfo()
            .setUsage(vma::MemoryUsage::eGpuOnly)
    );
}

static inline Buffer create_mapped_buffer(
    vk::Device device,
    vma::Allocator allocator,
    vk::BufferUsageFlags usage, 
    vk::DeviceSize size
) {
    return create_buffer(
        device,
        allocator,
        vk::BufferCreateInfo()
            .setSharingMode(vk::SharingMode::eExclusive)
            .setSize(size)
            .setUsage(usage | vk::BufferUsageFlagBits::eShaderDeviceAddress),
        vma::AllocationCreateInfo()
            .setFlags(
                vma::AllocationCreateFlagBits::eHostAccessSequentialWrite
                    | vma::AllocationCreateFlagBits::eMapped
            )
            .setUsage(vma::MemoryUsage::eAuto)
    );
}


static inline Buffer create_staging_buffer(
    vk::Device device,
    vma::Allocator allocator,
    vk::BufferUsageFlags usage, 
    vk::DeviceSize size
) {
    return create_buffer(
        device,
        allocator,
        vk::BufferCreateInfo()
            .setSharingMode(vk::SharingMode::eExclusive)
            .setSize(size)
            .setUsage(usage),
        vma::AllocationCreateInfo()
            .setFlags(
                vma::AllocationCreateFlagBits::eHostAccessSequentialWrite
                    | vma::AllocationCreateFlagBits::eMapped
            )
            .setUsage(vma::MemoryUsage::eAuto)
    );
}