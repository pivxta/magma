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
    this->buffer = Buffer(
        this->device,
        vk::BufferCreateInfo()
            .setUsage(usage | vk::BufferUsageFlagBits::eShaderDeviceAddress)
            .setSize(capacity_per_fif * static_cast<vk::DeviceSize>(frames_in_flight))
            .setSharingMode(vk::SharingMode::eExclusive),
        vma::AllocationCreateInfo()
            .setUsage(vma::MemoryUsage::eAuto)
            .setFlags(
                vma::AllocationCreateFlagBits::eMapped 
                | vma::AllocationCreateFlagBits::eHostAccessSequentialWrite
            )
    );
}

FrameArenaBuffer::~FrameArenaBuffer() {
    if (!this->device) {
        return;
    }
    this->buffer.destroy(this->device);
}

void FrameArenaBuffer::flush() {
    this->buffer.flush(this->device, 0, this->arena.used());
}

void FrameArenaBuffer::begin_frame(uint32_t next_frame_index) {
    this->frame_index = next_frame_index;
    this->arena.reset();
}