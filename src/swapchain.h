#pragma once
#include <memory>
#include <vulkan/vulkan.hpp>
#include "device.h"
#include "texture.h"

struct SwapchainConfigureInfo {
    vk::PresentModeKHR present_mode = vk::PresentModeKHR::eMailbox;
    vk::ColorSpaceKHR colorspace = vk::ColorSpaceKHR::eSrgbNonlinear;
    vk::Format format = vk::Format::eB8G8R8A8Srgb;

    SwapchainConfigureInfo& set_present_mode(vk::PresentModeKHR value) {
        this->present_mode = value;
        return *this;
    }

    SwapchainConfigureInfo& set_colorspace(vk::ColorSpaceKHR value) {
        this->colorspace = value;
        return *this;
    }

    SwapchainConfigureInfo& set_format(vk::Format value) {
        this->format = value;
        return *this;
    }
};

struct SwapchainTexture {
    uint32_t index;
    const Texture* texture;
    vk::Semaphore available;
    vk::Semaphore presentable;
};

class Target;

class Swapchain {
public:
    Swapchain() = default;
    Swapchain(
        const InstanceHandle& instance,
        const std::shared_ptr<Target>& target
    );
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&& other) noexcept;
    Swapchain& operator=(Swapchain&& other) noexcept;

    vk::SurfaceKHR surface() const {
        return this->surface_;
    }

    vk::Extent2D extent() const {
        return this->extent_;
    }

    vk::Format format() const {
        return this->format_;
    }

    vk::Viewport full_viewport() const {
        vk::Extent2D target_extent = this->extent();
        return vk::Viewport()
            .setX(0.0f)
            .setY(0.0f)
            .setWidth(static_cast<float>(target_extent.width))
            .setHeight(static_cast<float>(target_extent.height))
            .setMinDepth(0.0f)
            .setMaxDepth(1.0f);
    }

    vk::Rect2D full_area() const {
        return {{0, 0}, this->extent()};
    }
    
    vk::Result configure(const DeviceHandle& device, const SwapchainConfigureInfo& info);
    vk::Result present(const SwapchainTexture& texture);
    std::tuple<vk::Result, SwapchainTexture> acquire_texture(vk::Semaphore image_available);

private:
    InstanceHandle instance;
    DeviceHandle device;
    std::shared_ptr<Target> target;

    vk::Extent2D extent_;
    vk::Format format_ = vk::Format::eUndefined;

    vk::SurfaceKHR surface_;
    vk::SwapchainKHR swapchain;
    std::vector<Texture> textures;
    std::vector<vk::Semaphore> presentable;
};
