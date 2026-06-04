#include "swapchain.h"
#include "target.h"
#include "vkerror.h"

Swapchain::Swapchain(vk::Instance instance, const std::shared_ptr<Target>& target) {
    this->instance = instance;
    this->target = target;

    auto [result, surface] = target->target_surface(instance);
    vk_expect(result, "Failed to create surface");
    this->surface = surface;
}

static void destroy_semaphores(vk::Device device, std::vector<vk::Semaphore>& semaphores) {
    for (auto semaphore : semaphores) {
        if (semaphore != vk::Semaphore()) {
            device.destroySemaphore(semaphore);
            semaphore = vk::Semaphore();
        }
    }
}

static void recreate_semaphores(
    vk::Device device, 
    std::vector<vk::Semaphore>& semaphores, 
    size_t count
) {
    destroy_semaphores(device, semaphores);
    semaphores.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        auto [result, semaphore] = device.createSemaphore(vk::SemaphoreCreateInfo());
        vk_expect(result, "Failed to create semaphore");
        semaphores[i] = semaphore;
    }
}

static void destroy_views(vk::Device device, std::vector<vk::ImageView>& views) {
    for (auto& view : views) {
        if (view != vk::ImageView()) {
            device.destroyImageView(view);
            view = vk::ImageView();
        }
    }
}

static void recreate_views(
    vk::Device device,
    std::vector<vk::ImageView>& views,
    const std::vector<vk::Image>& images,
    vk::Format format
) {
    destroy_views(device, views);
    views.resize(images.size());
    for (size_t i = 0; i < views.size(); i++) {
        auto [result, view] = device.createImageView(
            vk::ImageViewCreateInfo()
                .setImage(images[i])
                .setFormat(format)
                .setComponents(vk::ComponentMapping())
                .setViewType(vk::ImageViewType::e2D)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setBaseMipLevel(0)
                        .setLevelCount(1)
                        .setBaseArrayLayer(0)
                        .setLayerCount(1)
                )
        );
        vk_expect(result, "Failed to create swapchain image view");
        views[i] = view;
    }
}

static vk::Extent2D pick_extent(
    const vk::SurfaceCapabilitiesKHR& caps,
    vk::Extent2D fallback_extent
) {
    if (caps.currentExtent.width != 0xffffffff) {
        return caps.currentExtent;
    }

    return {
        std::clamp(fallback_extent.width, caps.minImageExtent.width, caps.maxImageExtent.width),
        std::clamp(fallback_extent.height, caps.minImageExtent.height, caps.maxImageExtent.height)
    };
}

static vk::SurfaceFormatKHR pick_surface_format(
    vk::PhysicalDevice physical_device,
    vk::SurfaceKHR surface,
    vk::Format preferred_format,
    vk::ColorSpaceKHR preferred_colorspace
) {
    auto [result, surface_formats] = physical_device.getSurfaceFormatsKHR(surface);
    vk_expect(result, "Failed to get surface formats");

    vk::SurfaceFormatKHR surface_format = surface_formats[0];
    for (auto current_format : surface_formats) {
        if (current_format.format == preferred_format &&
            current_format.colorSpace == preferred_colorspace) {
            surface_format = current_format;
            break;
        }
    }

    return surface_format;
}

static vk::PresentModeKHR pick_present_mode(
    vk::PhysicalDevice physical_device, 
    vk::SurfaceKHR surface,
    vk::PresentModeKHR preferred
) {
    auto [result, present_modes] = physical_device.getSurfacePresentModesKHR(surface);
    vk_expect(result, "Failed to get surface present modes");

    vk::PresentModeKHR present_mode = vk::PresentModeKHR::eFifo;
    for (auto current_present_mode : present_modes) {
        if (current_present_mode == preferred) {
            present_mode = current_present_mode;
            break;
        }
    }

    return present_mode;
}

void Swapchain::configure(
    vk::PhysicalDevice physical_device, 
    vk::Device device, 
    const SwapchainConfigureInfo& info
) {
    auto [result, surface_caps] = physical_device.getSurfaceCapabilitiesKHR(this->surface);
    vk_expect(result, "Physical device surface capabilities not available");

    auto extent = pick_extent(surface_caps, this->target->target_extent());
    auto present_mode = pick_present_mode(physical_device, this->surface, info.present_mode);
    auto surface_format = pick_surface_format(
        physical_device, 
        this->surface, 
        info.format, 
        info.colorspace
    );

    uint32_t image_count = surface_caps.minImageCount + 1;
    if (surface_caps.maxImageCount > 0) {
        image_count = std::min(image_count, surface_caps.maxImageCount);
    }

    vk_expect(device.waitIdle(), "Failed to wait for device");
    auto [result1, swapchain] = device.createSwapchainKHR(
        vk::SwapchainCreateInfoKHR()
            .setSurface(this->surface)
            .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
            .setMinImageCount(image_count)
            .setImageExtent(extent)
            .setImageFormat(surface_format.format)
            .setImageColorSpace(surface_format.colorSpace)
            .setImageArrayLayers(1)
            .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
            .setPreTransform(surface_caps.currentTransform)
            .setPresentMode(present_mode)
            .setClipped(true)
            .setOldSwapchain(this->swapchain)
    );
    vk_expect(result1, "Failed to create swapchain");

    if (this->swapchain != vk::SwapchainKHR()) {
        device.destroySwapchainKHR(this->swapchain);
    }

    this->swapchain = swapchain;
    this->extent = extent;
    this->format = surface_format.format;

    auto [result2, swapchain_images] = device.getSwapchainImagesKHR(this->swapchain);
    vk_expect(result2, "Swapchain images not available");
    this->images = std::move(swapchain_images);
    recreate_views(device, this->views, this->images, this->format);
    recreate_semaphores(device, this->presentable, this->images.size());
}

std::tuple<vk::Result, SwapchainImage> Swapchain::acquire_image(
    vk::Device device, 
    vk::Semaphore image_available
) {
    auto [result, index] = device.acquireNextImageKHR(
        this->swapchain, std::numeric_limits<uint64_t>::max(), image_available
    );
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        return {result, SwapchainImage{}};
    }

    return {
        result,
        SwapchainImage{
            .index = index,
            .image = this->images[index],
            .view = this->views[index],
            .available = image_available,
            .presentable = this->presentable[index]
        }
    };
}

vk::Result Swapchain::present(vk::Queue queue, const SwapchainImage& image) {
    vk::Result result = queue.presentKHR(
        vk::PresentInfoKHR()
            .setSwapchains(this->swapchain)
            .setImageIndices(image.index)
            .setWaitSemaphores(image.presentable)
    );
    return result;
}

void Swapchain::destroy(vk::Device device) {
    vk_expect(device.waitIdle(), "Wait for device failed");
    destroy_semaphores(device, this->presentable);
    destroy_views(device, this->views);
    device.destroySwapchainKHR(this->swapchain);
    this->instance.destroySurfaceKHR(this->surface);
}