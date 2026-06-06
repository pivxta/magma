#include "framearena.h"

FrameArena::FrameArena(
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
    this->frames_in_flight = frames_in_flight;
    this->buffer = create_mapped_buffer(
        device, 
        allocator, 
        usage,
        capacity_per_fif * static_cast<vk::DeviceSize>(frames_in_flight)
    );
}

void FrameArena::destroy() {
    this->buffer.destroy(this->allocator);
}

void FrameArena::reset() {
    this->frame_index = (this->frame_index + 1) % this->frames_in_flight;
    this->arena.reset();
}