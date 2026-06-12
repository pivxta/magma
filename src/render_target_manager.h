#pragma once 
#include <vulkan/vulkan.hpp>
#include <variant>
#include <cstdint>
#include "texture.h"

struct RenderTargetId {
    uint32_t index = 0;
    uint32_t generation = 0;
};

enum class RenderTargetBuffering: uint8_t {
    Shared,
    PerFif
};

struct SwapchainAdjustedSizePolicy {
    double scale = 1.0;
};

struct FixedSizePolicy {
    vk::Extent2D extent;
};

using SizePolicy = std::variant<SwapchainAdjustedSizePolicy, FixedSizePolicy>;

struct RenderTargetInfo {
    RenderTargetBuffering buffering = RenderTargetBuffering::PerFif;
    SizePolicy size_policy = SwapchainAdjustedSizePolicy{};
    vk::ImageUsageFlags usage = {};
    vk::SampleCountFlags samples = vk::SampleCountFlagBits::e1;
    vk::Format format = vk::Format::eUndefined;
    uint32_t mip_levels = 1;
    uint32_t array_layers = 1;
};

struct RenderTargetSubresourceState {
    vk::ImageLayout layout;
    vk::AccessFlags2 access;
    vk::PipelineStageFlags2 stage;
};

struct RenderTargetUsage {
    uint32_t base_mip_level = 0;
    uint32_t mip_level_count = 1;
    uint32_t base_array_layer = 0;
    uint32_t array_layer_count = 1;
    RenderTargetSubresourceState new_state;
    bool discard = false;
};

class RenderTargetManager {
public:
    RenderTargetManager() = default;
    void destroy();
    void destroy_pending();

    RenderTargetId reserve(); 
    void bind(
        RenderTargetId id, 
        const Texture& texture, 
        RenderTargetSubresourceState state
    );

    RenderTargetId add(const RenderTargetInfo& info);
    const Texture* get(RenderTargetId id);
    void free(RenderTargetId id);

    RenderTargetSubresourceState state(
        RenderTargetId id, 
        uint32_t array_layer = 0, 
        uint32_t mip_level = 0
    );
    void use(RenderTargetId id, vk::CommandBuffer command_buffer, RenderTargetUsage uses);
    void use(
        RenderTargetId id,
        vk::CommandBuffer command_buffer,
        std::span<RenderTargetUsage> uses
    );
    
    void resize_swapchain(vk::Extent2D swapchain_extent);
    void begin_frame(uint64_t frame_counter);

private:
    struct Slot {
        RenderTargetBuffering buffering;
        SizePolicy size_policy;
        vk::SampleCountFlags samples;

        std::vector<vk::ImageView> views;
        std::vector<Texture> textures;
        bool owned;

        std::vector<RenderTargetSubresourceState> states;
    };
};