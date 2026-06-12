#pragma once
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include "image.h"
#include "texture.h"
#include "slot_map.h"
#include "resource.h"
#include "texture_indices.h"
#include "uploader.h"
#include "mip_map.h"
#include "bindless_set.h"

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
        const DeviceHandle& device,
        Uploader& uploader,
        uint32_t frames_in_flight,
        uint32_t max_textures,
        const GlobalSamplerInfo& sampler_info = {}
    );

    TextureManager(const TextureManager&) = delete;
    TextureManager& operator=(const TextureManager&) = delete;
    TextureManager(TextureManager&&) noexcept = default;
    TextureManager& operator=(TextureManager&&) noexcept = default;

    void update_pending();
    
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

    void configure_samplers(const GlobalSamplerInfo& info) {
        this->bindless_set.configure_samplers(info);
    }

    void begin_frame(uint64_t frame_counter) {
        this->bindless_set.begin_frame(frame_counter);
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
        return this->bindless_set.descriptor_set_layout();
    }
    
    vk::DescriptorSet descriptor_set() const {
        return this->bindless_set.descriptor_set();
    }
    
private:
    std::optional<SlotKey<Texture>> create_texture(Uploader& uploader, const Image& image);
    void create_fallbacks(Uploader& uploader);

    struct Slot {
        uint32_t sampler_index;
        std::optional<SlotKey<Texture>> texture;
        SlotKey<Texture> fallback;
    };

    static TextureId get_texture_id(SlotKey<Slot> key);
    static SlotKey<Slot> get_slot_key(TextureId id);
    static TextureIndices get_slot_indices(const Slot& slot);
    TextureIndices get_fallback_indices(TextureFallback fallback) const;

    using Fallbacks = std::array<SlotKey<Texture>, static_cast<size_t>(TextureFallback::Count)>;

    BindlessSet bindless_set;
    MipmapGenerator mipmap_generator;

    std::vector<TextureId> updated;
    SlotMap<Slot> slots;
    Fallbacks fallbacks;
};