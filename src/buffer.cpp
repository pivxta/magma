#include "buffer.h"
#include "vk_error.h"

Buffer::Buffer(
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

    this->buffer = buffer;
    this->allocation = allocation;
    this->size_ = buffer_info.size;
    this->address_ = address;
    this->mapped_data = info.pMappedData;
}