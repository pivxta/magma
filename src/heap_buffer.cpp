#include "heap_buffer.h"

HeapBuffer::HeapBuffer(
    vk::Device device, 
    vma::Allocator allocator,
    uint32_t frames_in_flight,
    vk::DeviceSize min_alignment,
    const vk::BufferCreateInfo& buffer_info,
    const vma::AllocationCreateInfo& alloc_info
):
    free_list(buffer_info.size, FreeListPolicy::FirstFit)
{
    this->allocator = allocator;
    this->frames_in_flight = frames_in_flight;
    this->min_alignment = min_alignment;
    this->buffer_ = create_buffer(
        device,
        allocator,
        buffer_info,
        alloc_info
    );
}

void HeapBuffer::destroy() {
    this->buffer_.destroy(this->allocator);
}

void HeapBuffer::begin_frame(uint64_t frame_counter) {
    this->frame_counter = frame_counter;
}

void HeapBuffer::free_pending() {
    while (!this->free_queue.empty()) {
        const PendingFree& pending = this->free_queue.back();
        if (frame_counter - pending.request_frame >= this->frames_in_flight) {
            this->free_list.free(pending.range);
            this->free_queue.pop_back();
        } else {
            break;
        }
    }
}