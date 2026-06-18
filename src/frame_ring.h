#pragma once
#include "device.h"

using FrameFn = std::function<void(vk::CommandBuffer)>;

class FrameRing {
public:
    FrameRing() = default;
    FrameRing(DeviceHandle device);
    ~FrameRing();

    FrameRing(const FrameRing&) = delete;
    FrameRing& operator=(const FrameRing&) = delete;
    FrameRing(FrameRing&&) noexcept = default;
    FrameRing& operator=(FrameRing&&) noexcept = default;

    [[nodiscard]]
    vk::Semaphore wait();

    void run(vk::Semaphore presentable, const FrameFn& frame_fn);

private:
    struct Frame {
        vk::Fence fence;
        vk::Semaphore available;
        vk::CommandBuffer command_buffer;
    };
    
    const Frame& current() const;
    
    DeviceHandle device;
    vk::CommandPool command_pool;
    std::vector<Frame> frames;
};