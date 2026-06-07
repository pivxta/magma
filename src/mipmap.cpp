#include "mipmap.h"

void MipmapGenerator::generate(Texture texture) {
    this->pending_gens.push_back(PendingGeneration{texture});
}

static void record_image_barrier(vk::CommandBuffer command_buffer, vk::ImageMemoryBarrier2 barrier) {
    command_buffer.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(barrier));
}

static void record_image_mipmap_gen(vk::CommandBuffer command_buffer, const Texture& texture) {
    for (uint32_t mip_level = 1; mip_level < texture.mip_levels; mip_level++) {
        record_image_barrier(
            command_buffer,
            vk::ImageMemoryBarrier2()
                .setImage(texture)
                .setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
                .setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite)
                .setDstStageMask(vk::PipelineStageFlagBits2::eBlit)
                .setDstAccessMask(vk::AccessFlagBits2::eTransferRead)
                .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
                .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setBaseMipLevel(mip_level - 1)
                        .setLevelCount(1)
                        .setBaseArrayLayer(0)
                        .setLayerCount(1)
                )
        );
        record_image_barrier(
            command_buffer,
            vk::ImageMemoryBarrier2()
                .setImage(texture)
                .setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
                .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                .setDstStageMask(vk::PipelineStageFlagBits2::eBlit)
                .setDstAccessMask(vk::AccessFlagBits2::eTransferWrite)
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setBaseMipLevel(mip_level)
                        .setLevelCount(1)
                        .setBaseArrayLayer(0)
                        .setLayerCount(1)
                )
        );

        auto src_width = int32_t(std::max(texture.extent.width >> (mip_level - 1), 1u));
        auto src_height = int32_t(std::max(texture.extent.height >> (mip_level - 1), 1u));
        auto dst_width = int32_t(std::max(texture.extent.width >> mip_level, 1u));
        auto dst_height = int32_t(std::max(texture.extent.height >> mip_level, 1u));

        std::array<vk::Offset3D, 2> src_offsets = {
            vk::Offset3D(0, 0, 0), 
            vk::Offset3D(src_width, src_height, 1)
        };
        auto src_subresource = vk::ImageSubresourceLayers()
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseArrayLayer(0)
            .setLayerCount(1)
            .setMipLevel(mip_level - 1);

        std::array<vk::Offset3D, 2> dst_offsets = {
            vk::Offset3D(0, 0, 0), 
            vk::Offset3D(dst_width, dst_height, 1)
        };
        auto dst_subresource = vk::ImageSubresourceLayers()
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseArrayLayer(0)
            .setLayerCount(1)
            .setMipLevel(mip_level);
        auto regions = vk::ImageBlit2()
            .setSrcSubresource(src_subresource)
            .setSrcOffsets(src_offsets)
            .setDstSubresource(dst_subresource)
            .setDstOffsets(dst_offsets);

        command_buffer.blitImage2(
            vk::BlitImageInfo2()
                .setSrcImage(texture)   
                .setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
                .setDstImage(texture)   
                .setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
                .setFilter(vk::Filter::eLinear)
                .setRegions(regions)
        );

        record_image_barrier(
            command_buffer,
            vk::ImageMemoryBarrier2()
                .setImage(texture)
                .setSrcStageMask(vk::PipelineStageFlagBits2::eBlit)
                .setSrcAccessMask(vk::AccessFlagBits2::eTransferRead)
                .setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
                .setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead)
                .setOldLayout(vk::ImageLayout::eTransferSrcOptimal)
                .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setBaseMipLevel(mip_level - 1)
                        .setLevelCount(1)
                        .setBaseArrayLayer(0)
                        .setLayerCount(1)
                )
        );
    }

    record_image_barrier(
        command_buffer,
        vk::ImageMemoryBarrier2()
            .setImage(texture)
            .setSrcStageMask(vk::PipelineStageFlagBits2::eBlit)
            .setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite)
            .setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
            .setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead)
            .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
            .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
            .setSubresourceRange(
                vk::ImageSubresourceRange()
                    .setAspectMask(vk::ImageAspectFlagBits::eColor)
                    .setBaseMipLevel(texture.mip_levels - 1)
                    .setLevelCount(1)
                    .setBaseArrayLayer(0)
                    .setLayerCount(1)
            )
    );
}

void MipmapGenerator::flush(vk::CommandBuffer command_buffer) {
    for (auto& generation: this->pending_gens) {
        record_image_mipmap_gen(command_buffer, generation.texture);
    }
    this->pending_gens.clear();
}