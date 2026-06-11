#pragma once
#include "image.h"
#include "texture.h"
#include "buffer.h"
#include "arena.h"

struct BufferUpload {
    const Buffer* buffer;
    vk::DeviceSize offset = 0;
    vk::AccessFlags2 usage_access = {};
    vk::PipelineStageFlags2 usage_stage = {};

    const void* memory = nullptr;
    size_t size = 0;

    BufferUpload& set_buffer(const Buffer& buffer) {
        this->buffer = &buffer;
        return *this;
    }

    BufferUpload& set_offset(vk::DeviceSize offset) {
        this->offset = offset;
        return *this;
    }

    BufferUpload& set_usage_access(vk::AccessFlags2 dst_access_mask) {
        this->usage_access = dst_access_mask;
        return *this;
    }

    BufferUpload& set_usage_stage(vk::PipelineStageFlags2 dst_stage_mask) {
        this->usage_stage = dst_stage_mask;
        return *this;
    }

    BufferUpload& set_memory(const void* memory, size_t size) {
        this->memory = memory;
        this->size = size;
        return *this;
    }

    template<typename T>
    BufferUpload& set_memory(std::span<const T> memory) {
        this->memory = memory.data();
        this->size = memory.size_bytes();
        return *this;
    }

    template<typename T>
    BufferUpload& set_memory(std::span<T> memory) {
        this->memory = memory.data();
        this->size = memory.size_bytes();
        return *this;
    }

    template<typename T>
    BufferUpload& set_memory(const std::vector<T>& memory) {
        this->memory = memory.data();
        this->size = memory.size() * sizeof(T);
        return *this;
    }
};

struct ImageUpload {
    const Texture* texture;
    const Image* image = nullptr;
    vk::Offset3D offset = vk::Offset3D(0, 0, 0);
    vk::Extent3D extent = vk::Extent3D(0, 0, 0);

    ImageUpload& set_texture(const Texture& texture) {
        this->texture = &texture;
        this->extent = texture.extent;
        return *this;
    }

    ImageUpload& set_image(const Image& image) {
        this->image = &image;
        return *this;
    }

    ImageUpload& set_offset(vk::Offset3D offset) {
        this->offset = offset;
        return *this;
    }

    ImageUpload& set_extent(vk::Extent3D extent) {
        this->extent = extent;
        return *this;
    }
};

class Uploader {
public:
    static constexpr vk::DeviceSize DEFAULT_ALIGNMENT = 256;

    Uploader() = default;
    Uploader(
        DeviceHandle device,
        uint32_t frames_in_flight,
        vk::DeviceSize capacity_per_fif
    );

    Uploader(const Uploader&) = delete;
    Uploader& operator=(const Uploader&) = delete;
    Uploader(Uploader&&) noexcept = default;
    Uploader& operator=(Uploader&&) noexcept = default;
    ~Uploader();

    bool upload_buffer(BufferUpload upload);
    bool upload_image(ImageUpload upload);

    // Called right after the frame counter is increased
    void begin_frame(uint32_t frame_index);
    void flush(vk::CommandBuffer command_buffer);

private:
    struct PendingImageUpload {
        Texture texture;
        vk::Extent3D extent;
        vk::Offset3D offset;
        ArenaAllocation<vk::DeviceSize> staging_range;

        uint32_t frame_index;
    };
    
    struct PendingBufferUpload {
        Buffer buffer;
        vk::DeviceSize size;
        vk::DeviceSize offset;
        vk::AccessFlags2 usage_access;
        vk::PipelineStageFlags2 usage_stage;
        ArenaAllocation<vk::DeviceSize> staging_range;

        uint32_t frame_index;
    };   

    void record_image_upload(vk::CommandBuffer command_buffer, const PendingImageUpload& upload);
    void record_buffer_upload(
        vk::CommandBuffer command_buffer, 
        const PendingBufferUpload& upload
    );

    DeviceHandle device;

    uint32_t frame_index = 0;

    Buffer buffer;
    vk::DeviceSize stride = 0;

    Arena<vk::DeviceSize> arena;
    std::vector<PendingImageUpload> pending_images;
    std::vector<PendingBufferUpload> pending_buffers;
};