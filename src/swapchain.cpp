#include "swapchain.h"
#include "target.h"
#include "vk_error.h"
#include <algorithm>
#include <limits>
#include <utility>

Swapchain::Swapchain(const InstanceHandle& instance, const std::shared_ptr<Target>& target) {
    this->instance = instance;
    this->target = target;

    auto [result, surface] = target->target_surface(instance->instance);
    vk_expect(result, "Failed to create surface");
    this->surface_ = surface;
}

Swapchain::Swapchain(Swapchain&& other) noexcept
    : instance(std::move(other.instance)),
      device(std::move(other.device)),
      target(std::move(other.target)),
      extent_(other.extent_),
      format_(other.format_),
      surface_(std::exchange(other.surface_, nullptr)),
      swapchain(std::exchange(other.swapchain, nullptr)),
      textures(std::move(other.textures)),
      presentable(std::move(other.presentable)) {}

Swapchain& Swapchain::operator=(Swapchain&& other) noexcept {
    if (this != &other) {
        std::swap(this->instance, other.instance);
        std::swap(this->device, other.device);
        std::swap(this->target, other.target);
        std::swap(this->extent_, other.extent_);
        std::swap(this->format_, other.format_);
        std::swap(this->surface_, other.surface_);
        std::swap(this->swapchain, other.swapchain);
        std::swap(this->textures, other.textures);
        std::swap(this->presentable, other.presentable);
    }
    return *this;
}

static void destroy_semaphores(vk::Device device, std::vector<vk::Semaphore>& semaphores) {
    for (auto& semaphore : semaphores) {
        if (semaphore) {
            device.destroySemaphore(semaphore);
            semaphore = nullptr;
        }
    }
}

static vk::Result recreate_semaphores(
    vk::Device device,
    std::vector<vk::Semaphore>& semaphores,
    size_t count
) {
    destroy_semaphores(device, semaphores);
    semaphores.resize(count);
    for (size_t i = 0; i < count; i++) {
        auto [result, semaphore] = device.createSemaphore(vk::SemaphoreCreateInfo());
        if (result != vk::Result::eSuccess) {
            return result;
        }
        semaphores[i] = semaphore;
    }
    return vk::Result::eSuccess;
}


static void destroy_textures(const DeviceHandle& device, std::vector<Texture>& textures) {
    for (auto& texture: textures) {
        texture.destroy(device);
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
    const std::vector<vk::SurfaceFormatKHR>& surface_formats,
    vk::Format preferred_format,
    vk::ColorSpaceKHR preferred_colorspace
) {
    for (auto current_format : surface_formats) {
        if (current_format.format == preferred_format &&
            current_format.colorSpace == preferred_colorspace) {
            return current_format;
        }
    }
    return surface_formats.front();
}

static vk::PresentModeKHR pick_present_mode(
    const std::vector<vk::PresentModeKHR>& present_modes,
    vk::PresentModeKHR preferred
) {
    for (auto current_present_mode : present_modes) {
        if (current_present_mode == preferred) {
            return current_present_mode;
        }
    }
    return vk::PresentModeKHR::eFifo;
}

vk::Result Swapchain::configure(const DeviceHandle& device, const SwapchainConfigureInfo& info) {
    this->device = device;

    auto [caps_result, surface_caps] = device->physical.getSurfaceCapabilitiesKHR(this->surface_);
    if (caps_result != vk::Result::eSuccess) {
        return caps_result;
    }

    auto [formats_result, surface_formats] =
        device->physical.getSurfaceFormatsKHR(this->surface_);
    if (formats_result != vk::Result::eSuccess) {
        return formats_result;
    }

    auto [modes_result, present_modes] =
        device->physical.getSurfacePresentModesKHR(this->surface_);
    if (modes_result != vk::Result::eSuccess) {
        return modes_result;
    }

    auto extent = pick_extent(surface_caps, this->target->target_extent());
    auto present_mode = pick_present_mode(present_modes, info.present_mode);
    auto surface_format = pick_surface_format(surface_formats, info.format, info.colorspace);

    uint32_t image_count = surface_caps.minImageCount + 1;
    if (surface_caps.maxImageCount > 0) {
        image_count = std::min(image_count, surface_caps.maxImageCount);
    }

    auto [create_result, new_swapchain] = device->logical.createSwapchainKHR(
        vk::SwapchainCreateInfoKHR()
            .setSurface(this->surface_)
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
    if (create_result != vk::Result::eSuccess) {
        return create_result;
    }

    destroy_textures(device, this->textures);
    if (this->swapchain) {
        device->logical.destroySwapchainKHR(this->swapchain);
    }
    this->swapchain = new_swapchain;
    this->extent_ = extent;
    this->format_ = surface_format.format;

    auto [images_result, new_images] =
        device->logical.getSwapchainImagesKHR(this->swapchain);
    if (images_result != vk::Result::eSuccess) {
        return images_result;
    }

    this->textures.resize(new_images.size());
    for (size_t i = 0; i < new_images.size(); i++) {
        this->textures[i] = Texture(
            device, new_images[i],
            vk::ImageType::e2D,
            this->format(),
            vk::Extent3D(this->extent(), 1),
            vk::ImageUsageFlagBits::eColorAttachment
        );
    }

    if (auto result = recreate_semaphores(device->logical, this->presentable, this->textures.size());
        result != vk::Result::eSuccess) 
    {
        return result;
    }

    return vk::Result::eSuccess;
}

std::tuple<vk::Result, SwapchainTexture> Swapchain::acquire_texture(vk::Semaphore image_available) {
    auto [result, index] = this->device->logical.acquireNextImageKHR(
        this->swapchain,
        std::numeric_limits<uint64_t>::max(),
        image_available
    );
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        return {result, SwapchainTexture{}};
    }

    return {
        result,
        SwapchainTexture{
            .index = index,
            .texture = &this->textures[index],
            .available = image_available,
            .presentable = this->presentable[index]
        }
    };
}

vk::Result Swapchain::present(const SwapchainTexture& image) {
    return this->device->graphics_queue.presentKHR(
        vk::PresentInfoKHR()
            .setSwapchains(this->swapchain)
            .setImageIndices(image.index)
            .setWaitSemaphores(image.presentable)
    );
}

Swapchain::~Swapchain() {
    if (this->device) {
        this->device->wait_idle();
        destroy_semaphores(this->device->logical, this->presentable);
        destroy_textures(this->device, this->textures);
        if (this->swapchain) {
            this->device->logical.destroySwapchainKHR(this->swapchain);
        }
    }
    // The surface belongs to the instance (created in the ctor, before any
    // device exists), so it is torn down via the instance, not the device.
    if (this->instance && this->surface_) {
        this->instance->instance.destroySurfaceKHR(this->surface_);
    }
}
