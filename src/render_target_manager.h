#pragma once 
#include <vulkan/vulkan.hpp>
#include <variant>
#include <cstdint>
#include "slot_map.h"
#include "texture.h"
#include "bindless_set.h"

struct RenderTargetId {
    uint32_t index = 0;
    uint32_t generation = 0;
};

struct RenderTargetIndices {
    uint32_t texture;
    uint32_t sampler;
};

enum class RenderTargetBuffering: uint8_t {
    Shared,
    PerFif
};

struct SwapchainAdjustedSizePolicy {
    double scale = 1.0;

    SwapchainAdjustedSizePolicy(double scale = 1.0): scale(scale) {}
};

struct FixedSizePolicy {
    vk::Extent2D extent;

    FixedSizePolicy(vk::Extent2D extent): extent(extent) {}
};

using SizePolicy = std::variant<SwapchainAdjustedSizePolicy, FixedSizePolicy>;

struct RenderTargetInfo {
    RenderTargetBuffering buffering = RenderTargetBuffering::PerFif;
    SizePolicy size_policy = SwapchainAdjustedSizePolicy{};
    vk::ImageUsageFlags usage = {}; // Required format-derived usage flags are set automatically
    vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
    vk::Format format = vk::Format::eUndefined;
    uint32_t mip_levels = 1;
    uint32_t array_layers = 1;

    RenderTargetInfo& set_buffering(RenderTargetBuffering value) {
        this->buffering = value;
        return *this;
    }

    RenderTargetInfo& set_size_policy(SizePolicy value) {
        this->size_policy = value;
        return *this;
    }

    RenderTargetInfo& set_usage(vk::ImageUsageFlags value) {
        this->usage = value;
        return *this;
    }

    RenderTargetInfo& set_samples(vk::SampleCountFlagBits value) {
        this->samples = value;
        return *this;
    }

    RenderTargetInfo& set_format(vk::Format value) {
        this->format = value;
        return *this;
    }

    RenderTargetInfo& set_mip_levels(uint32_t value) {
        this->mip_levels = value;
        return *this;
    }

    RenderTargetInfo& set_array_layers(uint32_t value) {
        this->array_layers = value;
        return *this;
    }
};

struct RenderTargetSubresourceState {
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    vk::AccessFlags2 access = vk::AccessFlagBits2::eNone;
    vk::PipelineStageFlags2 stage = vk::PipelineStageFlagBits2::eNone;

    RenderTargetSubresourceState& set_layout(vk::ImageLayout value) {
        this->layout = value;
        return *this;
    }

    RenderTargetSubresourceState& set_access(vk::AccessFlags2 value) {
        this->access = value;
        return *this;
    }

    RenderTargetSubresourceState& set_stage(vk::PipelineStageFlags2 value) {
        this->stage = value;
        return *this;
    }

    bool operator==(const RenderTargetSubresourceState&) const = default;
};

struct RenderTargetUsage {
    uint32_t base_mip_level = 0;
    uint32_t mip_level_count = 1;
    uint32_t base_array_layer = 0;
    uint32_t array_layer_count = 1;
    RenderTargetSubresourceState new_state;
    bool discard = false;

    RenderTargetUsage& set_base_mip_level(uint32_t value) {
        this->base_mip_level = value;
        return *this;
    }

    RenderTargetUsage& set_mip_level_count(uint32_t value) {
        this->mip_level_count = value;
        return *this;
    }

    RenderTargetUsage& set_mip_level_range(uint32_t base, uint32_t count) {
        this->base_mip_level = base;
        this->mip_level_count = count;
        return *this;
    }

    RenderTargetUsage& set_base_array_layer(uint32_t value) {
        this->base_array_layer = value;
        return *this;
    }

    RenderTargetUsage& set_array_layer_count(uint32_t value) {
        this->array_layer_count = value;
        return *this;
    }

    RenderTargetUsage& set_array_layer_range(uint32_t base, uint32_t count) {
        this->base_array_layer = base;
        this->array_layer_count = count;
        return *this;
    }

    RenderTargetUsage& set_new_state(RenderTargetSubresourceState value) {
        this->new_state = value;
        return *this;
    }

    RenderTargetUsage& set_discard(bool value = true) {
        this->discard = value;
        return *this;
    }
};

class RenderTargetManager {
public:
    RenderTargetManager() = default;
    RenderTargetManager(
        const DeviceHandle& device, 
        uint32_t frames_in_flight,
        uint32_t max_targets
    );

    RenderTargetManager(const RenderTargetManager&) = delete;
    const RenderTargetManager& operator=(const RenderTargetManager&) = delete;
    RenderTargetManager(RenderTargetManager&&) noexcept = default;
    RenderTargetManager& operator=(RenderTargetManager&&) noexcept = default;

    void destroy_pending() {
        this->bindless_set.update_pending();
    }

    std::optional<RenderTargetId> add(const RenderTargetInfo& info);
    std::optional<RenderTargetIndices> 
    get_indices(RenderTargetId id, Sampler sampler = Sampler::LinearRepeat);
    const Texture* get(RenderTargetId id);
    void free(RenderTargetId id);

    std::optional<RenderTargetId> reserve(); 

    bool bind(
        RenderTargetId id, 
        const Texture& texture, 
        RenderTargetSubresourceState state = {}
    );

    void use(
        vk::CommandBuffer command_buffer, 
        RenderTargetId id, 
        RenderTargetUsage usage
    );

    void use(
        vk::CommandBuffer command_buffer,
        RenderTargetId id,
        std::initializer_list<RenderTargetUsage> usages
    );

    void use(
        vk::CommandBuffer command_buffer,
        RenderTargetId id,
        std::span<const RenderTargetUsage> usages
    );

    RenderTargetSubresourceState state(
        RenderTargetId id, 
        uint32_t array_layer = 0, 
        uint32_t mip_level = 0
    );

    vk::DescriptorSet descriptor_set() const {
        return this->bindless_set.descriptor_set();
    }

    vk::DescriptorSetLayout descriptor_set_layout() const {
        return this->bindless_set.descriptor_set_layout();
    }
    
    void resize_swapchain(vk::Extent2D swapchain_extent);
    void begin_frame(uint64_t frame_counter);

private:
    struct Target {
        RenderTargetBuffering buffering = RenderTargetBuffering::Shared;
        std::optional<SizePolicy> size_policy;
        std::vector<RenderTargetSubresourceState> states;
        std::vector<SlotKey<Texture>> keys;
        std::vector<const Texture*> textures;
        uint32_t mip_levels = 0;
        uint32_t array_layers = 0;
        bool owned = false;
    };

    static SlotKey<Target> get_slot_key(RenderTargetId id);
    static RenderTargetId get_target_id(SlotKey<Target> key);
    size_t get_state_index(const Target& target, uint32_t array_layer, uint32_t mip_level);
    bool create_textures(Target& target, const RenderTargetInfo& info);
    bool recreate_textures(Target& target);
    const Texture* get_target_texture(const Target& target);

    BindlessSet bindless_set;
    SlotMap<Target> targets;
    vk::Extent2D swapchain_extent;
    uint32_t frames_in_flight;
    uint32_t frame_index = 0;
};