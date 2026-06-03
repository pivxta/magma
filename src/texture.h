#pragma once
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include "vkerror.h"

struct Texture {
    vk::Image image;
    vma::Allocation allocation;
    vk::ImageView view;

    uint32_t mip_levels;
    vk::Extent3D extent;
    vk::Format format;

    operator vk::Image() const {
        return this->image;
    }

    bool is_null() const {
        return this->image == vk::Image();
    }

    void destroy(vk::Device device, vma::Allocator allocator) {
        device.destroyImageView(this->view);
        allocator.destroyImage(this->image, this->allocation);
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

static inline Texture create_texture(
    vk::Device device,
    vma::Allocator allocator,
    const vk::ImageCreateInfo& info
) {
    auto alloc_info = vma::AllocationCreateInfo().setUsage(vma::MemoryUsage::eGpuOnly);
    auto [result1, resources] = allocator.createImage(info, alloc_info);
    vk_expect(result1, "Failed to create image");
    auto [alloc, image] = resources;

    auto [result2, view] = device.createImageView(
        vk::ImageViewCreateInfo()
            .setImage(image)
            .setFormat(info.format)
            .setComponents(vk::ComponentMapping())
            .setViewType(vk::ImageViewType::e2D)
            .setSubresourceRange(
                vk::ImageSubresourceRange()
                    .setAspectMask(get_default_aspect_flags(info.format))
                    .setBaseArrayLayer(0)
                    .setLayerCount(1)
                    .setBaseMipLevel(0)
                    .setLevelCount(info.mipLevels)
            )
    );
    vk_expect(result2, "Failed to create image view");

    return {
        .image = resources.second,
        .allocation = resources.first,
        .view = view,
        .mip_levels = info.mipLevels,
        .extent = info.extent,
        .format = info.format,
    };
}