#include "frame_ring.h"
#include "vk_error.h"

vk::CommandPool create_command_pool(const DeviceHandle& device) {
    auto [result, command_pool] = device->logical.createCommandPool(
        vk::CommandPoolCreateInfo()
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
            .setQueueFamilyIndex(device->graphics_queue_family)
    );
    vk_expect(result, "Failed to create command pool");
    return command_pool;
}

std::vector<vk::CommandBuffer> create_command_buffers(
    const DeviceHandle& device, 
    vk::CommandPool command_pool
) {
    auto [result, command_buffers] = device->logical.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo()
            .setCommandPool(command_pool)
            .setCommandBufferCount(device->frames_in_flight)
            .setLevel(vk::CommandBufferLevel::ePrimary)
    );
    vk_expect(result, "Failed to allocate command buffers");
    return command_buffers;
}

vk::Fence create_fence(const DeviceHandle& device) {
    auto [result, fence] = device->logical.createFence(
        vk::FenceCreateInfo().setFlags(vk::FenceCreateFlagBits::eSignaled)
    );
    vk_expect(result, "Failed to create in-flight fence");
    return fence;
}

vk::Semaphore create_semaphore(const DeviceHandle& device) {
    auto [result, semaphore] = device->logical.createSemaphore(vk::SemaphoreCreateInfo());
    vk_expect(result, "Failed to create image available fence");
    return semaphore;
}

FrameRing::FrameRing(DeviceHandle device) {
    this->device = std::move(device);
    this->command_pool = create_command_pool(this->device);
    std::vector command_buffers = create_command_buffers(this->device, this->command_pool);
    for (uint32_t i = 0; i < this->device->frames_in_flight; i++) {
        this->frames.push_back({
            .fence = create_fence(this->device),
            .available = create_semaphore(this->device),
            .command_buffer = command_buffers[i]
        });
    }
}

FrameRing::~FrameRing() {
    if (this->device != nullptr) {
        this->device->wait_idle();
        this->device->logical.destroyCommandPool(this->command_pool);
        this->command_pool = vk::CommandPool();
        for (auto& frame: this->frames) {
            this->device->logical.destroyFence(frame.fence);
            this->device->logical.destroySemaphore(frame.available);
            frame.available = vk::Semaphore();
            frame.fence = vk::Fence();
        }
    }
}

vk::Semaphore FrameRing::wait() {
    const auto& frame = this->frames[this->device->frame_index()];
    const auto fence = frame.fence;
    vk_expect(
        this->device->logical.waitForFences(
            fence, 
            true,
            std::numeric_limits<uint64_t>::max()
        ),
        "Fence wait failed"
    );
    return frame.available;
}

void FrameRing::run(vk::Semaphore presentable, const FrameFn& frame_fn) {
    const auto& frame = this->frames[this->device->frame_index()];
    const auto fence = frame.fence;
    vk_expect(this->device->logical.resetFences(fence), "Fence reset failed");
    this->device->flush_deferred();

    vk::CommandBuffer command_buffer = frame.command_buffer;
    vk_expect(command_buffer.reset(), "Failed to reset command buffer");
    vk_expect(
        command_buffer.begin(
            vk::CommandBufferBeginInfo()
                .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
        ),
        "Failed to begin command buffer"
    );

    frame_fn(command_buffer);

    vk_expect(command_buffer.end(), "Failed to end command buffer");

    auto command_buffer_info = vk::CommandBufferSubmitInfo().setCommandBuffer(command_buffer);

    auto wait_info = vk::SemaphoreSubmitInfo()
        .setSemaphore(frame.available)
        .setStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);

    auto signal_info = vk::SemaphoreSubmitInfo()
        .setSemaphore(presentable)
        .setStageMask(vk::PipelineStageFlagBits2::eAllCommands);

    vk_expect(
        this->device->graphics_queue.submit2(
            vk::SubmitInfo2()
                .setWaitSemaphoreInfos(wait_info)
                .setSignalSemaphoreInfos(signal_info)
                .setCommandBufferInfos(command_buffer_info),
            frame.fence
        ),
        "Failed to submit command buffer"
    );

    this->device->next_frame();
}
