#include "texture.h"
#include "vk_error.h"

static vk::ImageViewType get_default_view_type(vk::ImageType type, uint32_t array_layers) {
    assert(array_layers >= 1);
    if (array_layers > 1) {
        switch (type) {
            case vk::ImageType::e1D: return vk::ImageViewType::e1DArray;
            case vk::ImageType::e2D: return vk::ImageViewType::e2DArray;
            case vk::ImageType::e3D: assert(false && "Cannot create 3D texture arrays");
        }
    } else {
        switch (type) {
            case vk::ImageType::e1D: return vk::ImageViewType::e1D;
            case vk::ImageType::e2D: return vk::ImageViewType::e2D;
            case vk::ImageType::e3D: return vk::ImageViewType::e3D;
        }
    }
    return vk::ImageViewType::e2D;
}

Texture::Texture(const DeviceHandle& device, const vk::ImageCreateInfo& info) {
    auto alloc_info = vma::AllocationCreateInfo().setUsage(vma::MemoryUsage::eGpuOnly);
    auto [result1, resources] = device->allocator.createImage(info, alloc_info);
    vk_expect(result1, "Failed to create image");

    this->owned = true;
    this->image = resources.second;
    this->allocation = resources.first;
    this->type_ = info.imageType;
    this->array_layers_ = info.arrayLayers;
    this->mip_levels_ = info.mipLevels;
    this->samples_ = info.samples;
    this->extent_ = info.extent;
    this->format_ = info.format;
    this->usage_ = info.usage;
    this->view_ = this->create_view(
        device, 
        TextureViewCreateInfo()
            .set_base_mip_level(0)
            .set_mip_level_count(this->mip_levels())
            .set_base_array_layer(0)
            .set_array_layer_count(this->array_layers())
    );
}

Texture::Texture(
    const DeviceHandle& device,
    vk::Image image,
    vk::ImageType type,
    vk::Format format,
    vk::Extent3D extent,
    vk::ImageUsageFlags usage,
    uint32_t array_layers,
    uint32_t mip_levels,
    vk::SampleCountFlagBits samples
) {
    this->owned = false;
    this->image = image;
    this->type_ = type;
    this->format_ = format;
    this->extent_ = extent;
    this->usage_ = usage;
    this->array_layers_ = array_layers;
    this->mip_levels_ = mip_levels;
    this->samples_ = samples;
    this->view_ = this->create_view(
        device, 
        TextureViewCreateInfo()
            .set_base_mip_level(0)
            .set_mip_level_count(mip_levels)
            .set_base_array_layer(0)
            .set_array_layer_count(array_layers)
    );
}

vk::ImageView Texture::create_view(
    const DeviceHandle& device, 
    const TextureViewCreateInfo& info // Aspect flags set automatically if not already set
) const {
    auto [result, image_view] = device->logical.createImageView(
        vk::ImageViewCreateInfo()
            .setImage(this->image)
            .setFormat(this->format())
            .setViewType(get_default_view_type(this->type(), info.array_layer_count))
            .setSubresourceRange(
                vk::ImageSubresourceRange()
                    .setAspectMask(get_default_aspect_flags(this->format()))
                    .setBaseArrayLayer(info.base_array_layer)
                    .setLayerCount(info.array_layer_count)
                    .setBaseMipLevel(info.base_mip_level)
                    .setLevelCount(info.mip_level_count)
            ) 
    );
    vk_expect(result, "Failed to create image view");
    return image_view;
}