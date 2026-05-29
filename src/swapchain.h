#pragma once
#include <memory>
#include <vulkan/vulkan.hpp>

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

struct SwapchainImage {
    uint32_t index;
    vk::Image image;
    vk::ImageView view;
    vk::Semaphore available;
    vk::Semaphore presentable;
};

class Target;

class Swapchain {
public:
    Swapchain();
    Swapchain(vk::Instance instance, const std::shared_ptr<Target>& target);
    ~Swapchain();

    void destroy(vk::Device device);

    vk::SurfaceKHR get_surface();
    vk::Extent2D get_extent() const;
    vk::Format get_format() const;
    vk::Viewport get_full_viewport() const;
    vk::Rect2D get_full_area() const;
    
    void configure(
        vk::PhysicalDevice physical_device, 
        vk::Device device, 
        const SwapchainConfigureInfo& info
    );

    std::tuple<vk::Result, SwapchainImage> acquire_image(
        vk::Device device, 
        vk::Fence fence, 
        vk::Semaphore image_available
    );

    vk::Result present(vk::Queue queue, const SwapchainImage& image);

private:
    struct Inner;
    std::shared_ptr<Inner> inner;
};
