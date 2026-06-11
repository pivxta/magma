#include "buffer.h"
#include "vk_error.h"

Buffer create_buffer(
    const DeviceHandle& device,
    vk::BufferCreateInfo buffer_info,
    const vma::AllocationCreateInfo& alloc_info
) {
    vma::AllocationInfo info;
    auto [result, resources] = device->allocator.createBuffer(buffer_info, alloc_info, &info);
    vk_expect(result, "Failed to create buffer");

    vk::Buffer buffer = resources.second;
    vma::Allocation allocation = resources.first;
    vk::DeviceAddress address = 0;
    if (buffer_info.usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
        address = device->logical
            .getBufferAddress(vk::BufferDeviceAddressInfo().setBuffer(buffer));
    }

    return {
        .buffer = buffer,
        .allocation = allocation,
        .size = buffer_info.size,
        .address = address,
        .mapped_data = info.pMappedData
    };
}

Buffer create_gpu_buffer(
    const DeviceHandle& device,
    vk::BufferUsageFlags usage, 
    vk::DeviceSize size
) {
    return create_buffer(
        device,
        vk::BufferCreateInfo()
            .setSharingMode(vk::SharingMode::eExclusive)
            .setSize(size)
            .setUsage(usage | vk::BufferUsageFlagBits::eShaderDeviceAddress),
        vma::AllocationCreateInfo()
            .setUsage(vma::MemoryUsage::eGpuOnly)
    );
}

Buffer create_mapped_buffer(
    const DeviceHandle& device,
    vk::BufferUsageFlags usage, 
    vk::DeviceSize size
) {
    return create_buffer(
        device,
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


Buffer create_staging_buffer(
    const DeviceHandle& device,
    vk::BufferUsageFlags usage, 
    vk::DeviceSize size
) {
    return create_buffer(
        device,
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