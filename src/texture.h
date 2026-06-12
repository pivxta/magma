#pragma once
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include "device.h"

class Texture {
public:
    Texture() = default;
    explicit Texture(const DeviceHandle& device, const vk::ImageCreateInfo& info);

    void destroy(const DeviceHandle& device) {
        device->logical.destroyImageView(this->view_);
        device->allocator.destroyImage(this->image, this->allocation);
    }

    vk::ImageView default_view() const {
        return this->view_;
    }
    
    vk::Extent3D extent() const {
        return this->extent_;
    }

    vk::Format format() const {
        return this->format_;
    }

    vk::SampleCountFlags samples() const {
        return this->samples_;
    }

    uint32_t array_layers() const {
        return this->array_layers_;
    }

    uint32_t mip_levels() const {
        return this->mip_levels_;
    }

    bool is_null() const {
        return this->image == vk::Image();
    }

    operator vk::Image() const {
        return this->image;
    }

private:
    vk::Image image;
    vma::Allocation allocation;
    vk::ImageView view_;

    uint32_t array_layers_ = 0;
    uint32_t mip_levels_ = 0;
    vk::SampleCountFlags samples_ = {};
    vk::Extent3D extent_;
    vk::Format format_ = vk::Format::eUndefined;
};

static inline vk::ImageAspectFlags get_default_aspect_flags(vk::Format format) {
    switch (format) {
        case vk::Format::eD16Unorm:
        case vk::Format::eX8D24UnormPack32:
        case vk::Format::eD32Sfloat:
            return vk::ImageAspectFlagBits::eDepth;

        case vk::Format::eS8Uint:
            return vk::ImageAspectFlagBits::eStencil;

        case vk::Format::eD16UnormS8Uint:
        case vk::Format::eD24UnormS8Uint:
        case vk::Format::eD32SfloatS8Uint:
            return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;

        default:
            return vk::ImageAspectFlagBits::eColor;
    }
}
