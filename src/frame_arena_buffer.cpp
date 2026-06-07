#include "frame_arena_buffer.h"

FrameArenaBuffer::FrameArenaBuffer(
    vk::Device device, 
    vma::Allocator allocator, 
    vk::BufferUsageFlags usage,
    uint32_t frames_in_flight,
    vk::DeviceSize capacity_per_fif
): 
    arena(capacity_per_fif)
{
    this->allocator = allocator;
    this->stride = capacity_per_fif;
    this->buffer = create_mapped_buffer(
        device, 
        allocator, 
        usage,
        capacity_per_fif * static_cast<vk::DeviceSize>(frames_in_flight)
    );
}

void FrameArenaBuffer::destroy() {
    this->buffer.destroy(this->allocator);
}

void FrameArenaBuffer::begin_frame(uint32_t next_frame_index) {
    this->frame_index = next_frame_index;
    this->arena.reset();
}