#pragma once
#include <optional>
#include <memory>
#include <functional>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include "target.h"

class Swapchain;
struct Instance;
struct Device;
using DeviceHandle = std::shared_ptr<const Device>;
using InstanceHandle = std::shared_ptr<const Instance>;

struct Instance {
    vk::Instance instance;
    vk::DebugUtilsMessengerEXT debug_messenger = nullptr;

    ~Instance();

    static InstanceHandle create(const Target& target);
};

struct Device {
    InstanceHandle instance;
    vk::PhysicalDeviceProperties2 properties;
    vk::PhysicalDevice physical;
    vma::Allocator allocator;
    vk::Device logical;
    vk::Queue graphics_queue;
    uint32_t graphics_queue_family = 0;
    uint32_t frames_in_flight;

    static std::optional<DeviceHandle> create(
        const InstanceHandle& instance, 
        const Swapchain& swapchain,
        uint32_t frames_in_flight
    );

    ~Device();

    void defer(std::function<void()> fn) const;
    void next_frame() const;
    void flush_deferred() const;
    void wipe_deferred() const;
    uint64_t frame_counter() const;
    uint32_t frame_index() const;

    void wait_idle() const;

private:
    struct DeferredQueue;
    struct FrameState;

    std::unique_ptr<DeferredQueue> deferred_queue;
    std::unique_ptr<FrameState> frame_state;
};

