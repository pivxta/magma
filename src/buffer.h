#pragma once
#include <vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include <span>
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

template<typename T>
struct DynOffsetBuffer {
    Buffer buffer;
    vk::DeviceSize length;
    vk::DeviceSize count;
    uint32_t stride;

    operator vk::Buffer() const {
        return this->buffer.buffer;
    }

    uint32_t offset(uint32_t frame_index) const {
        return static_cast<uint32_t>(this->stride) 
            * static_cast<uint32_t>(frame_index);;
    }

    T* get_mapped(uint32_t frame_index) {
        return reinterpret_cast<T*>(
            reinterpret_cast<uint8_t*>(this->buffer.mapped) 
                + static_cast<size_t>(this->stride) * static_cast<size_t>(frame_index)
        );
    }
    
    void write_all(std::span<const T> data) {
        for (uint32_t i = 0; i < this->count; i++) {
            this->write_one(i, data);
        }
    }

    void write_all(const T& data) {
        for (uint32_t i = 0; i < this->count; i++) {
            this->write_one(i, data);
        }
    }

    void write_one(uint32_t frame_index, std::span<const T> data) {
        size_t write_size = data.size_bytes();
        if (write_size > this->stride) {
            write_size = this->stride;
            spdlog::warn("Write size of dynamic offset buffer higher than the stride");
        }
        std::memcpy(this->get_mapped(frame_index), data.data(), write_size);
    }
    
    void write_one(uint32_t frame_index, const T& data) {
        std::memcpy(this->get_mapped(frame_index), &data, sizeof(T));
    }

    void destroy(vma::Allocator allocator) {
        this->buffer.destroy(allocator);
        this->count = 0;
        this->length = 0;
    }

    vk::DescriptorBufferInfo descriptor_info(
        vk::DeviceSize offset = 0,
        std::optional<vk::DeviceSize> elems = std::nullopt
    ) {
        vk::DeviceSize length = elems.value_or(this->length);
        return vk::DescriptorBufferInfo()
            .setBuffer(*this)
            .setOffset(offset)
            .setRange(sizeof(T) * length);
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

template<typename T>
DynOffsetBuffer<T> create_dyn_offset_buffer(
    vk::Device device, 
    vma::Allocator allocator,
    vk::BufferUsageFlags usage, 
    vk::DeviceSize length,
    vk::DeviceSize count
) {
    auto props = allocator.getPhysicalDeviceProperties();
    vk::DeviceSize align = std::max(
        props->limits.minStorageBufferOffsetAlignment,
        props->limits.minUniformBufferOffsetAlignment
    );
    vk::DeviceSize stride = (length * sizeof(T) + align - 1) & ~(align - 1);
    Buffer buffer = create_buffer(
        device,
        allocator,
        vk::BufferCreateInfo()
            .setSharingMode(vk::SharingMode::eExclusive)
            .setSize(stride * count)
            .setUsage(usage),
        vma::AllocationCreateInfo()
            .setFlags(
                vma::AllocationCreateFlagBits::eHostAccessSequentialWrite
                    | vma::AllocationCreateFlagBits::eMapped
            )
            .setUsage(vma::MemoryUsage::eAuto)
    );
    return {
        .buffer = buffer, 
        .length = length, 
        .count = count, 
        .stride = static_cast<uint32_t>(stride)
    };
}
