#include "heap_buffer.h"

HeapBuffer::HeapBuffer(
    DeviceHandle device,
    vk::DeviceSize min_alignment,
    const vk::BufferCreateInfo& buffer_info,
    const vma::AllocationCreateInfo& alloc_info
):
    device(std::move(device))
{
    this->min_alignment = min_alignment;
    this->buffer_ = Buffer(this->device, buffer_info, alloc_info);
    this->free_list = std::make_shared<FreeList<vk::DeviceSize>>(
        buffer_info.size, 
        FreeListPolicy::FirstFit
    );
}

HeapBuffer::~HeapBuffer() {
    if (this->device != nullptr) {
        this->device->wait_idle();
        this->buffer_.destroy(this->device);
    }
}