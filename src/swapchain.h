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
    Swapchain() = default;
    Swapchain(vk::Instance instance, const std::shared_ptr<Target>& target);

    void destroy(vk::Device device);

    vk::SurfaceKHR get_surface() {
        return this->surface;
    }

    vk::Extent2D get_extent() const {
        return this->extent;
    }

    vk::Format get_format() const {
        return this->format;
    }

    vk::Viewport get_full_viewport() const {
        vk::Extent2D target_extent = this->get_extent();
        return vk::Viewport()
            .setX(0.0f)
            .setY(0.0f)
            .setWidth(static_cast<float>(target_extent.width))
            .setHeight(static_cast<float>(target_extent.height))
            .setMinDepth(0.0f)
            .setMaxDepth(1.0f);
    }

    vk::Rect2D get_full_area() const {
        return {{0, 0}, this->get_extent()};
    }
    
    void configure(
        vk::PhysicalDevice physical_device, 
        vk::Device device, 
        const SwapchainConfigureInfo& info
    );

    std::tuple<vk::Result, SwapchainImage> acquire_image(
        vk::Device device, 
        vk::Semaphore image_available
    );

    vk::Result present(vk::Queue queue, const SwapchainImage& image);

private:
    vk::Instance instance;
    std::shared_ptr<Target> target;

    vk::Extent2D extent;
    vk::Format format = vk::Format::eUndefined;
    std::vector<vk::Image> images;

    vk::SurfaceKHR surface;
    vk::SwapchainKHR swapchain;
    std::vector<vk::ImageView> views;
    std::vector<vk::Semaphore> presentable;
};
