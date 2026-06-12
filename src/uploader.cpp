#include "uploader.h"
#include <spdlog/spdlog.h>

Uploader::Uploader(
    DeviceHandle device,
    uint32_t frames_in_flight,
    vk::DeviceSize capacity_per_fif
):
    device(std::move(device)),
    arena(capacity_per_fif)
{
    this->stride = capacity_per_fif;
    this->buffer = Buffer(
        this->device,
        vk::BufferCreateInfo()
            .setSharingMode(vk::SharingMode::eExclusive)
            .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
            .setSize(capacity_per_fif * frames_in_flight),
        vma::AllocationCreateInfo()
            .setUsage(vma::MemoryUsage::eAuto)
            .setFlags(
                vma::AllocationCreateFlagBits::eMapped
                | vma::AllocationCreateFlagBits::eHostAccessSequentialWrite
            )
    );
}

Uploader::~Uploader() {
    if (!this->device) {
        return;
    }
    this->buffer.destroy(this->device);
}

bool Uploader::upload_buffer(BufferUpload upload) {
    assert(upload.buffer != nullptr);

    if (upload.size + upload.offset > upload.buffer->size()) {
        spdlog::error("Buffer not big enough for upload");
        return false;
    }
    
    if (upload.memory == nullptr || upload.size == 0) {
        spdlog::error("Buffer upload call provided invalid source memory");
        return false;
    }

    Buffer buffer = *upload.buffer;
    if (buffer.mapped() != nullptr) {
        memcpy(buffer.mapped(upload.offset), upload.memory, upload.size);
        buffer.flush(this->device, upload.offset, upload.size);
        return true;
    }

    auto staging_range = this->arena.allocate(upload.size, DEFAULT_ALIGNMENT);
    if (!staging_range.has_value()) {
        return false;
    }

    vk::DeviceSize buffer_offset = this->stride * this->frame_index + staging_range->offset;
    memcpy(this->buffer.mapped(buffer_offset), upload.memory, upload.size);
    this->buffer.flush(this->device, buffer_offset, upload.size);

    this->pending_buffers.push_back(PendingBufferUpload{
        .buffer = buffer,
        .size = upload.size,
        .offset = upload.offset,
        .usage_access = upload.usage_access,
        .usage_stage = upload.usage_stage,
        .staging_range = staging_range.value(),
        .frame_index = this->frame_index
    });
    return true;
}

bool Uploader::upload_image(ImageUpload upload) {
    assert(upload.texture != nullptr);
    assert(upload.image != nullptr);

    if (upload.image->bytes.size() < upload.image->expected_size_bytes()) {
        spdlog::error("Image has less bytes than expected for upload.");
        return false;
    }

    vk::DeviceSize size = upload.image->expected_size_bytes();
    auto staging_range = this->arena.allocate(size, DEFAULT_ALIGNMENT);
    if (!staging_range.has_value()) {
        return false;
    }

    vk::DeviceSize buffer_offset = this->stride * this->frame_index + staging_range->offset;
    memcpy(this->buffer.mapped(buffer_offset), upload.image->bytes.data(), size);
    this->buffer.flush(this->device, buffer_offset, size);

    this->pending_images.push_back(PendingImageUpload{
        .texture = *upload.texture,
        .extent = upload.extent,
        .offset = upload.offset,
        .staging_range = staging_range.value(),
        .frame_index = this->frame_index
    });
    return true;
}

static void record_buffer_barrier(
    vk::CommandBuffer command_buffer, 
    vk::BufferMemoryBarrier2 barrier
) {
    command_buffer.pipelineBarrier2(vk::DependencyInfo().setBufferMemoryBarriers(barrier));
}

static void record_image_barrier(vk::CommandBuffer command_buffer, vk::ImageMemoryBarrier2 barrier) {
    command_buffer.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(barrier));
}

void Uploader::record_buffer_upload(
    vk::CommandBuffer command_buffer, 
    const PendingBufferUpload& upload
) {
    command_buffer.copyBuffer(
        this->buffer,
        upload.buffer,
        {
            vk::BufferCopy()
                .setSrcOffset(this->stride * upload.frame_index + upload.staging_range.offset)
                .setDstOffset(upload.offset)
                .setSize(upload.size)
        }
    );

    record_buffer_barrier(
        command_buffer,
        vk::BufferMemoryBarrier2()
            .setBuffer(upload.buffer)
            .setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
            .setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite)
            .setDstStageMask(upload.usage_stage)
            .setDstAccessMask(upload.usage_access)
            .setOffset(upload.offset)
            .setSize(upload.size)
    );
}

void Uploader::record_image_upload(
    vk::CommandBuffer command_buffer, 
    const PendingImageUpload& upload
) {
    auto subresource_range = vk::ImageSubresourceRange()
        .setAspectMask(get_default_aspect_flags(upload.texture.format()))
        .setBaseMipLevel(0)
        .setLevelCount(1)
        .setBaseArrayLayer(0)
        .setLayerCount(1);

    record_image_barrier(
        command_buffer,
        vk::ImageMemoryBarrier2()
            .setImage(upload.texture) 
            .setSrcStageMask(vk::PipelineStageFlagBits2::eNone)
            .setSrcAccessMask(vk::AccessFlagBits2::eNone)
            .setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
            .setDstAccessMask(vk::AccessFlagBits2::eTransferWrite)
            .setOldLayout(vk::ImageLayout::eUndefined)
            .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
            .setSubresourceRange(subresource_range)
    );
    
    command_buffer.copyBufferToImage(
        this->buffer,
        upload.texture, 
        vk::ImageLayout::eTransferDstOptimal,
        {
            vk::BufferImageCopy()
                .setBufferOffset(this->stride * upload.frame_index + upload.staging_range.offset)
                .setImageOffset(upload.offset)
                .setImageExtent(upload.extent)
                .setImageSubresource(
                    vk::ImageSubresourceLayers()
                        .setAspectMask(get_default_aspect_flags(upload.texture.format()))
                        .setBaseArrayLayer(0)
                        .setLayerCount(1)
                        .setMipLevel(0)
                )
        }
    );
}

void Uploader::begin_frame(uint32_t frame_index) {
    this->frame_index = frame_index;
    this->arena.reset();
}

void Uploader::flush(vk::CommandBuffer command_buffer) {
    for (auto& upload: this->pending_images) {
        this->record_image_upload(command_buffer, upload);
    }
    this->pending_images.clear();

    for (auto& upload: this->pending_buffers) {
        this->record_buffer_upload(command_buffer, upload);
    }
    this->pending_buffers.clear();
}