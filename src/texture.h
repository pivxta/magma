#pragma once
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include "device.h"

struct TextureViewCreateInfo {
    uint32_t base_array_layer = 0;
    uint32_t array_layer_count = 1;
    uint32_t base_mip_level = 0;
    uint32_t mip_level_count = 1;

    TextureViewCreateInfo& set_base_array_layer(uint32_t base_array_layer) {
        this->base_array_layer = base_array_layer;
        return *this;
    }

    TextureViewCreateInfo& set_array_layer_count(uint32_t array_layer_count) {
        this->array_layer_count = array_layer_count;
        return *this;
    }

    TextureViewCreateInfo& set_base_mip_level(uint32_t base_mip_level) {
        this->base_mip_level = base_mip_level;
        return *this;
    }

    TextureViewCreateInfo& set_mip_level_count(uint32_t mip_level_count) {
        this->mip_level_count = mip_level_count;
        return *this;
    }
};

class Texture {
public:
    Texture() = default;

    Texture(const Texture&) noexcept = default;
    Texture& operator=(const Texture&) noexcept = default;
    Texture(Texture&&) noexcept = default;
    Texture& operator=(Texture&&) noexcept = default;
    
    // Owned texture constructor.
    explicit Texture(const DeviceHandle& device, const vk::ImageCreateInfo& info);
    
    // Borrowed texture constructor, only the view is owned. Mainly used for the swapchain.
    explicit Texture(
        const DeviceHandle& device,
        vk::Image image,
        vk::ImageType type,
        vk::Format format,
        vk::Extent3D extent,
        vk::ImageUsageFlags usage = {},
        uint32_t array_layers = 1,
        uint32_t mip_levels = 1,
        vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1
    );

    void destroy(const DeviceHandle& device) {
        if (this->view_ != vk::ImageView{}) {
            device->logical.destroyImageView(this->view_);
            this->view_ = vk::ImageView{};
        }
        if (this->owned && this->image != vk::Image{}) {
            device->allocator.destroyImage(this->image, this->allocation);
            this->image = vk::Image{};
        }
    }

    vk::ImageView create_view(
        const DeviceHandle& handle, 
        const TextureViewCreateInfo& info
    ) const;

    vk::ImageView default_view() const {
        return this->view_;
    }

    vk::ImageUsageFlags usage() const {
        return this->usage_;
    }

    vk::ImageType type() const {
        return this->type_;
    }
    
    vk::Extent3D extent() const {
        return this->extent_;
    }

    vk::Format format() const {
        return this->format_;
    }

    vk::SampleCountFlagBits samples() const {
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
    bool owned = false;

    vk::Image image;
    vma::Allocation allocation;
    vk::ImageView view_;

    uint32_t array_layers_ = 0;
    uint32_t mip_levels_ = 0;
    vk::SampleCountFlagBits samples_ = vk::SampleCountFlagBits::e1;
    vk::Extent3D extent_;
    vk::Format format_ = vk::Format::eUndefined;
    vk::ImageType type_ = vk::ImageType::e1D;
    vk::ImageUsageFlags usage_ = {};
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
