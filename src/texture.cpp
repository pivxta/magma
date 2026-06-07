#include "texture.h"
#include "vk_error.h"

Texture create_texture(
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