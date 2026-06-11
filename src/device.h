#pragma once
#include <optional>
#include <memory>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include "target.h"

class Swapchain;

struct InstanceContext {
    vk::Instance instance;
    vk::DebugUtilsMessengerEXT debug_messenger = nullptr;

    ~InstanceContext();
};

using InstanceHandle = std::shared_ptr<const InstanceContext>;

struct DeviceContext {
    InstanceHandle instance;
    vk::PhysicalDeviceProperties2 properties;
    vk::PhysicalDevice physical;
    vk::Device logical;
    vk::Queue graphics_queue;
    uint32_t graphics_queue_family = 0;
    vma::Allocator allocator;

    ~DeviceContext();

    void wait_idle() const;
};

using DeviceHandle = std::shared_ptr<const DeviceContext>;

InstanceHandle create_instance(const Target& target);
std::optional<DeviceHandle> create_device(
    const InstanceHandle& instance, 
    const Swapchain& swapchain
);