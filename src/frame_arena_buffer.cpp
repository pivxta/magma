#include "frame_arena_buffer.h"

FrameArenaBuffer::FrameArenaBuffer(
    DeviceHandle device,
    vk::BufferUsageFlags usage,
    uint32_t frames_in_flight,
    vk::DeviceSize capacity_per_fif
): 
    device(std::move(device)),
    arena(capacity_per_fif)
{
    this->stride = capacity_per_fif;
    this->buffer = create_mapped_buffer(
        this->device, 
        usage,
        capacity_per_fif * static_cast<vk::DeviceSize>(frames_in_flight)
    );
}

FrameArenaBuffer::~FrameArenaBuffer() {
    if (!this->device) {
        return;
    }
    this->buffer.destroy(this->device);
}

void FrameArenaBuffer::begin_frame(uint32_t next_frame_index) {
    this->frame_index = next_frame_index;
    this->arena.reset();
}