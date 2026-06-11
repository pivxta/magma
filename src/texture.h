#pragma once
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include "device.h"

struct Texture {
    vk::Image image;
    vma::Allocation allocation;
    vk::ImageView view;

    uint32_t array_layers = 0;
    uint32_t mip_levels = 0;
    vk::SampleCountFlags samples = {};
    vk::Extent3D extent;
    vk::Format format = vk::Format::eUndefined;

    bool is_null() const {
        return this->image == vk::Image();
    }

    void destroy(const DeviceHandle& device) {
        device->logical.destroyImageView(this->view);
        device->allocator.destroyImage(this->image, this->allocation);
    }

    operator vk::Image() const {
        return this->image;
    }
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

Texture create_texture(const DeviceHandle& device, const vk::ImageCreateInfo& info);