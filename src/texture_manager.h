#pragma once
#include <cstdint>
#include <vector>
#include <span>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include "image.h"
#include "texture.h"
#include "slot_map.h"
#include "resource.h"
#include "texture_indices.h"
#include "uploader.h"
#include "mip_map.h"

enum class TextureFallback: uint8_t {
    ColorError,
    ColorWhite,
    Normal,
    Displacement,
    AoRoughnessMetallic,
    Count,
};

class TextureManager {
public:
    TextureManager() = default;

    TextureManager(
        vk::Device device, 
        vma::Allocator allocator,
        Uploader& uploader,
        uint32_t frames_in_flight,
        uint32_t max_textures
    );
    void destroy_pending();
    void destroy();
    
    TextureIndices get(TextureId id, TextureFallback fallback = TextureFallback::ColorError) const;
    TextureIndices get_fallback(TextureFallback fallback = TextureFallback::ColorError) const;

    TextureId reserve(TextureFallback fallback = TextureFallback::ColorError);
    TextureId add(
        Uploader& uploader,
        const Image& image,
        TextureFallback fallback = TextureFallback::ColorError
    );
    void set(
        TextureId id,
        Uploader& uploader,
        const Image& image,
        TextureFallback fallback = TextureFallback::ColorError
    );
    void free(TextureId id);

    void begin_frame(uint64_t frame_counter) {
        this->frame_counter = frame_counter;
    }

    void clear_updated() {
        this->updated.clear();
    }

    void flush_mip_maps(vk::CommandBuffer command_buffer) {
        this->mipmap_generator.flush(command_buffer);
    }

    const std::vector<TextureId>& get_updated() const {
        return this->updated;
    }

    vk::DescriptorSetLayout descriptor_set_layout() const {
        return this->desc_set_layout;
    }
    
    vk::DescriptorSet descriptor_set() const {
        return this->desc_set;
    }
    
private:
    void create_fallbacks(Uploader& uploader);
    void create_samplers(vk::Device device);
    void bind_samplers(vk::Device device, vk::DescriptorSet set, std::span<Sampler> types);
    void destroy_samplers(vk::Device device);

    std::optional<SlotKey<Texture>> create_texture(Uploader& uploader, const Image& image);
    void destroy_texture(SlotKey<Texture> key);

    struct Slot {
        uint32_t sampler_index;
        std::optional<SlotKey<Texture>> texture;
        SlotKey<Texture> fallback;
    };

    static TextureId get_texture_id(SlotKey<Slot> key);
    static SlotKey<Slot> get_slot_key(TextureId id);
    static TextureIndices get_slot_indices(const Slot& slot);
    TextureIndices get_fallback_indices(TextureFallback fallback) const;

    struct PendingDestroy {
        uint64_t request_frame;
        SlotKey<Texture> texture;
    };

    using Samplers = std::array<vk::Sampler, static_cast<size_t>(Sampler::Count)>;
    using Fallbacks = std::array<SlotKey<Texture>, static_cast<size_t>(TextureFallback::Count)>;

    vk::Device device;
    vma::Allocator allocator;

    vk::DescriptorPool desc_pool;
    vk::DescriptorSetLayout desc_set_layout;
    vk::DescriptorSet desc_set;

    MipmapGenerator mipmap_generator;

    std::vector<TextureId> updated;
    SlotMap<Slot> slots;
    SlotMap<Texture> textures;
    Samplers samplers;
    Fallbacks fallbacks;
    std::deque<PendingDestroy> destroy_queue;
    uint32_t frames_in_flight;
    uint64_t frame_counter = 0;
};