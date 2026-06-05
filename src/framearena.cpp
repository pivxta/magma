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

std::optional<FrameSubBuffer> FrameArena::allocate(
    vk::DeviceSize size, 
    vk::DeviceSize aligmnent
) {
    auto alloc = this->arena.allocate(size, aligmnent);
    if (!alloc.has_value()) {
        return std::nullopt;
    }

    vk::DeviceSize offset = this->stride * this->frame_index + alloc->offset;
    return FrameSubBuffer{
        .parent_allocation = this->buffer.allocation,
        .base_address = this->buffer.address + offset,
        .base_offset = offset,
        .mapped = this->buffer.get_mapped(offset),
        .local_range = *alloc,
    };
}

void FrameArena::reset() {
    this->frame_index = (this->frame_index + 1) % this->frames_in_flight;
    this->arena.reset();
}