#include "frame_arena_buffer.h"

FrameArenaBuffer::FrameArenaBuffer(
    DeviceHandle device,
    vk::BufferUsageFlags usage,
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
            .setSize(capacity_per_fif * static_cast<vk::DeviceSize>(this->device->frames_in_flight))
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
    if (this->device != nullptr) {
        this->device->wait_idle();
        this->buffer.destroy(this->device);
    }
}

void FrameArenaBuffer::flush() {
    this->buffer.flush(this->device, 0, this->arena.used());
}

void FrameArenaBuffer::reset() {
    this->arena.reset();
}