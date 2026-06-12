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
}

Texture::Texture(const DeviceHandle& device, const vk::ImageCreateInfo& info) {
    auto alloc_info = vma::AllocationCreateInfo().setUsage(vma::MemoryUsage::eGpuOnly);
    auto [result1, resources] = device->allocator.createImage(info, alloc_info);
    vk_expect(result1, "Failed to create image");
    auto [alloc, image] = resources;

    auto [result2, view] = device->logical.createImageView(
        vk::ImageViewCreateInfo()
            .setImage(image)
            .setFormat(info.format)
            .setComponents(vk::ComponentMapping())
            .setViewType(get_default_view_type(info.imageType, info.arrayLayers))
            .setSubresourceRange(
                vk::ImageSubresourceRange()
                    .setAspectMask(get_default_aspect_flags(info.format))
                    .setBaseArrayLayer(0)
                    .setLayerCount(info.arrayLayers)
                    .setBaseMipLevel(0)
                    .setLevelCount(info.mipLevels)
            )
    );
    vk_expect(result2, "Failed to create image view");

    this->image = resources.second;
    this->allocation = resources.first;
    this->view_ = view;
    this->array_layers_ = info.arrayLayers;
    this->mip_levels_ = info.mipLevels;
    this->samples_ = info.samples;
    this->extent_ = info.extent;
    this->format_ = info.format;
}