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

    FixedSizePolicy(uint32_t width, uint32_t height): extent(vk::Extent2D(width, height)) {}
    FixedSizePolicy(vk::Extent2D extent): extent(extent) {}
};

using SizePolicy = std::variant<SwapchainAdjustedSizePolicy, FixedSizePolicy>;

struct RenderTargetInfo {
    SizePolicy size_policy = SwapchainAdjustedSizePolicy{};
    vk::ImageUsageFlags usage = {}; // Required format-derived usage flags are set automatically
    vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
    vk::Format format = vk::Format::eUndefined;
    uint32_t mip_levels = 1;
    uint32_t array_layers = 1;

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

struct MipLevelRange {
    uint32_t base = 0;
    uint32_t count = 1;

    MipLevelRange(uint32_t base, uint32_t count) {
        this->base = base;
        this->count = count;
    }
};

struct ArrayLayerRange {
    uint32_t base = 0;
    uint32_t count = 1;

    ArrayLayerRange(uint32_t base, uint32_t count) {
        this->base = base;
        this->count = count;
    }
};

struct RenderTargetUsage {
    std::optional<ArrayLayerRange> array_layers;
    std::optional<MipLevelRange> mip_levels;
    RenderTargetSubresourceState new_state;
    bool discard = false;

    static RenderTargetUsage color_attachment() {
        return RenderTargetUsage().set_new_state(
            RenderTargetSubresourceState()
                .set_layout(vk::ImageLayout::eColorAttachmentOptimal)
                .set_stage(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
                .set_access(vk::AccessFlagBits2::eColorAttachmentWrite)
        ).set_discard(true);
    }

    static RenderTargetUsage depth_attachment() {
        return RenderTargetUsage().set_new_state(
            RenderTargetSubresourceState()
                .set_layout(vk::ImageLayout::eDepthAttachmentOptimal)
                .set_stage(
                    vk::PipelineStageFlagBits2::eEarlyFragmentTests
                        | vk::PipelineStageFlagBits2::eLateFragmentTests
                )
                .set_access(
                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                        | vk::AccessFlagBits2::eDepthStencilAttachmentRead
                )
        ).set_discard(true);
    }

    static RenderTargetUsage shader_read() {
        return RenderTargetUsage().set_new_state(
            RenderTargetSubresourceState()
                .set_layout(vk::ImageLayout::eShaderReadOnlyOptimal)
                .set_stage(vk::PipelineStageFlagBits2::eFragmentShader)
                .set_access(vk::AccessFlagBits2::eShaderSampledRead)
        );
    }

    static RenderTargetUsage present() {
        return RenderTargetUsage().set_new_state(
            RenderTargetSubresourceState()
                .set_layout(vk::ImageLayout::ePresentSrcKHR)
                .set_stage(vk::PipelineStageFlagBits2::eNone)
                .set_access(vk::AccessFlagBits2::eNone)
        );
    }

    RenderTargetUsage& set_mip_level_range(uint32_t base, uint32_t count) {
        this->mip_levels = MipLevelRange(base, count);
        return *this;
    }

    RenderTargetUsage& set_array_layer_range(uint32_t base, uint32_t count) {
        this->array_layers = ArrayLayerRange(base, count);
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

class Swapchain;

class RenderTargetManager {
public:
    RenderTargetManager() = default;
    RenderTargetManager(
        const DeviceHandle& device, 
        const Swapchain& swapchain,
        uint32_t max_targets
    );

    RenderTargetManager(const RenderTargetManager&) = delete;
    const RenderTargetManager& operator=(const RenderTargetManager&) = delete;
    RenderTargetManager(RenderTargetManager&&) noexcept = default;
    RenderTargetManager& operator=(RenderTargetManager&&) noexcept = default;

    std::optional<RenderTargetId> reserve(); 
    std::optional<RenderTargetId> add(const RenderTargetInfo& info);
    bool reset(RenderTargetId id, const RenderTargetInfo& info);
    void bind(RenderTargetId id, const Texture& texture);
    void free(RenderTargetId id);
    
    const Texture* get_texture(RenderTargetId id) const;
    std::optional<RenderTargetIndices> get_indices(
        RenderTargetId id, 
        Sampler sampler = Sampler::LinearRepeat
    ) const;
    std::optional<RenderTargetIndices> get_indices(
        RenderTargetId id, 
        ComparisonSampler sampler
    ) const;

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
    ) const;

    vk::DescriptorSet descriptor_set() const {
        return this->bindless_set.descriptor_set();
    }

    vk::DescriptorSetLayout descriptor_set_layout() const {
        return this->bindless_set.descriptor_set_layout();
    }
    
    void resize_swapchain(vk::Extent2D swapchain_extent);

private:
    struct Target {
        bool owned = false;
        const Texture* bound = nullptr;
        RenderTargetBuffering buffering = RenderTargetBuffering::Shared;
        std::vector<RenderTargetSubresourceState> states;
        std::optional<SizePolicy> size_policy;
        std::optional<BindlessKey> key;
        uint32_t mip_levels = 0;
        uint32_t array_layers = 0;
    };

    static SlotKey<Target> get_slot_key(RenderTargetId id);
    static RenderTargetId get_target_id(SlotKey<Target> key);
    std::optional<Target> create_target(const RenderTargetInfo& info);
    bool initialize_target(Target& target, const RenderTargetInfo& info);
    bool recreate_target(Target& target);
    const Texture* get_target_texture(const Target& target) const;

    BindlessSet bindless_set;
    SlotMap<Target> targets;
    vk::Extent2D swapchain_extent;
};