#pragma once
#include <cstdint>
#include <vector>
#include <span>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include "image.h"
#include "texture.h"
#include "slotmap.h"
#include "resource.h"
#include "textureindices.h"

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
        vk::Queue queue,
        vma::Allocator allocator,
        vk::CommandPool command_pool,
        uint32_t frames_in_flight,
        uint32_t max_textures
    );
    void destroy_pending(uint64_t frame_counter);
    void destroy();
    
    TextureIndices get(TextureId id, TextureFallback fallback = TextureFallback::ColorError) const;
    TextureIndices get_fallback(TextureFallback fallback = TextureFallback::ColorError) const;

    TextureId reserve(TextureFallback fallback = TextureFallback::ColorError);
    TextureId add(
        vk::Queue queue,
        vk::CommandPool pool,
        const Image& image,
        TextureFallback fallback = TextureFallback::ColorError
    );
    void set(
        TextureId id,
        vk::Queue queue,
        vk::CommandPool pool,
        uint64_t frame_counter,
        const Image& image,
        TextureFallback fallback = TextureFallback::ColorError
    );
    void request_free(TextureId id, uint64_t frame_counter);

    void clear_updated() {
        this->updated.clear();
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
    void create_fallbacks(vk::Queue queue, vk::CommandPool command_pool);
    void create_samplers(vk::Device device);
    void bind_samplers(vk::Device device, vk::DescriptorSet set, std::span<Sampler> types);
    void destroy_samplers(vk::Device device);

    std::optional<SlotKey<Texture>> create_texture(
        vk::Queue queue, 
        vk::CommandPool pool, 
        const Image& image
    );
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

    std::vector<TextureId> updated;
    SlotMap<Slot> slots;
    SlotMap<Texture> textures;
    Samplers samplers;
    Fallbacks fallbacks;
    std::deque<PendingDestroy> destroy_queue;
    uint32_t frames_in_flight;
};