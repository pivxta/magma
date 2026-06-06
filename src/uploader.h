#pragma once
#include "image.h"
#include "texture.h"
#include "buffer.h"
#include "arena.h"

class Uploader {
public:
    Uploader();
    void destroy();

    bool upload_buffer(const Buffer& buffer, void* memory, size_t size, vk::DeviceSize offset);

    template<typename T>
    bool upload_buffer(const Buffer& buffer, std::span<T> memory, vk::DeviceSize offset) {
        return this->upload_buffer(buffer, memory.data(), memory.size(), offset);
    }

    bool upload_image(
        const Texture& texture, 
        const Image& image, 
        vk::Offset2D offset = vk::Offset2D(0, 0)
    );

    void flush();

private:
    struct PendingImageUpload {
        Texture texture;
        vk::Offset2D offset;
        ArenaAllocation<vk::DeviceSize> staging_range;
    };
    
    struct PendingBufferUpload {
        Buffer buffer;
        vk::DeviceSize offset;
        ArenaAllocation<vk::DeviceSize> staging_range;
    };   

    vma::Allocator allocator;

    uint32_t frame_index = 0;
    uint32_t frames_in_flight = 0;

    Buffer buffer;
    vk::DeviceSize stride = 0;

    Arena<vk::DeviceSize> arena;
    std::vector<PendingImageUpload> pending_images;
    std::vector<PendingBufferUpload> pending_buffers;
};