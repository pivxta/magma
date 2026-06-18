#include "render_target_manager.h"
#include "swapchain.h"
#include <cmath>
#include <spdlog/spdlog.h>

RenderTargetManager::RenderTargetManager(
    const DeviceHandle& device, 
    const Swapchain& swapchain,
    uint32_t max_targets
): 
    bindless_set(device, max_targets),
    targets(max_targets)
{
    this->resize_swapchain(swapchain.extent());
}

SlotKey<RenderTargetManager::Target> RenderTargetManager::get_slot_key(RenderTargetId id) {
    return {
        .index = id.index,
        .generation = id.generation
    };
}

RenderTargetId RenderTargetManager::get_target_id(SlotKey<Target> key) {
    return {
        .index = key.index,
        .generation = key.generation
    };
}

std::optional<RenderTargetId> RenderTargetManager::reserve() {
    if (auto key = this->targets.insert({}); key.has_value()) {
        return get_target_id(key.value());
    }
    return std::nullopt;
}
    
bool RenderTargetManager::reset(RenderTargetId id, const RenderTargetInfo& info) {
    if (auto target = this->targets.get(get_slot_key(id)); target != nullptr) {
        return this->initialize_target(*target, info);
    }
    return false;
}

void RenderTargetManager::bind(RenderTargetId id, const Texture& texture) {
    if (auto target = this->targets.get(get_slot_key(id)); target != nullptr) {
        assert(!target->owned && "Cannot bind texture to owned target.");
        target->bound = &texture;
        target->mip_levels = texture.mip_levels();
        target->array_layers = texture.array_layers();
        target->states.clear();
        target->states.resize(static_cast<size_t>(target->array_layers) * target->mip_levels, {});
    }
}

std::optional<RenderTargetId> RenderTargetManager::add(const RenderTargetInfo& info) {
    const auto id = this->reserve();
    if (!id.has_value()) {
        return std::nullopt;
    }
    if (!this->reset(*id, info)) {
        this->free(*id);
        return std::nullopt;
    }
    return id;
}

std::optional<RenderTargetIndices> RenderTargetManager::get_indices(
    RenderTargetId id,
    Sampler sampler
) const {
    if (auto target = this->targets.get(get_slot_key(id)); target != nullptr) {
        assert(target->owned);
        return RenderTargetIndices{
            .texture = target->key->index,
            .sampler = this->bindless_set.get_sampler(sampler)
        };
    }
    return std::nullopt;
}

std::optional<RenderTargetIndices> RenderTargetManager::get_indices(
    RenderTargetId id,
    ComparisonSampler sampler
) const {
    if (auto target = this->targets.get(get_slot_key(id)); target != nullptr) {
        return RenderTargetIndices{
            .texture = target->key->index,
            .sampler = this->bindless_set.get_sampler(sampler)
        };
    }
    return std::nullopt;
}

const Texture* RenderTargetManager::get_texture(RenderTargetId id) const {
    if (auto target = this->targets.get(get_slot_key(id)); target != nullptr) {
        return this->get_target_texture(*target);
    }
    return nullptr;
}

void RenderTargetManager::free(RenderTargetId id) {
    this->targets.free(get_slot_key(id), [&](Target& target) {
        if (target.key.has_value()) {
            this->bindless_set.free_texture(target.key.value());
        }
        target.key = std::nullopt;
        target.states = {};
    });
}

RenderTargetSubresourceState RenderTargetManager::state(
    RenderTargetId id, 
    uint32_t array_layer,
    uint32_t mip_level
) const {
    if (auto target = this->targets.get(get_slot_key(id)); target != nullptr) {
        return target->states[array_layer * target->mip_levels + mip_level];
    }
    return {};
}

void RenderTargetManager::use(
    vk::CommandBuffer command_buffer, 
    RenderTargetId id, 
    RenderTargetUsage usage
) {
    this->use(command_buffer, id, {usage});
}

void RenderTargetManager::use(
    vk::CommandBuffer command_buffer,
    RenderTargetId id,
    std::initializer_list<RenderTargetUsage> usages
) {
    this->use(command_buffer, id, std::span(usages));
}

void RenderTargetManager::use(
    vk::CommandBuffer command_buffer,
    RenderTargetId id,
    std::span<const RenderTargetUsage> usages
) {
    constexpr size_t MAX_BARRIERS = 128; 
    if (auto target = this->targets.get(get_slot_key(id)); target != nullptr) {
        size_t barrier_count = 0;
        auto texture = this->get_target_texture(*target);
        auto aspect = texture->used_aspects();
        std::array<vk::ImageMemoryBarrier2, MAX_BARRIERS> barriers;
        for (const auto& usage: usages) {
            uint32_t base_array_layer = 0;
            uint32_t array_layer_count = texture->array_layers();
            uint32_t base_mip_level = 0;
            uint32_t mip_level_count = texture->mip_levels();

            if (usage.mip_levels.has_value()) {
                base_mip_level = usage.mip_levels->base;
                mip_level_count = usage.mip_levels->count;
            }

            if (usage.array_layers.has_value()) {
                base_array_layer = usage.array_layers->base;
                array_layer_count = usage.array_layers->count;
            }
            
            for (uint32_t layer_offset = 0; layer_offset < array_layer_count; layer_offset++) {
                for (uint32_t mip_offset = 0; mip_offset < mip_level_count; mip_offset++) {
                    uint32_t layer = base_array_layer + layer_offset;
                    uint32_t level = base_mip_level + mip_offset;
                    auto& state = target->states[layer * target->mip_levels + level];

                    RenderTargetSubresourceState src_state = usage.discard ?
                        RenderTargetSubresourceState{} :
                        state;
                    
                    assert(barrier_count < MAX_BARRIERS && "Too many render target usages");
                    barriers[barrier_count] = vk::ImageMemoryBarrier2()
                        .setSrcAccessMask(src_state.access)
                        .setSrcStageMask(src_state.stage)
                        .setOldLayout(src_state.layout)
                        .setDstAccessMask(usage.new_state.access)
                        .setDstStageMask(usage.new_state.stage)
                        .setNewLayout(usage.new_state.layout)
                        .setImage(*texture)
                        .setSubresourceRange(
                            vk::ImageSubresourceRange()
                                .setAspectMask(aspect)
                                .setBaseArrayLayer(layer)
                                .setLayerCount(1)
                                .setBaseMipLevel(level)
                                .setLevelCount(1)
                        );

                    state = usage.new_state;
                    barrier_count += 1;
                }
            }
        }

        if (barrier_count > 0) {
            command_buffer.pipelineBarrier2(
                vk::DependencyInfo()
                    .setPImageMemoryBarriers(barriers.data())
                    .setImageMemoryBarrierCount(static_cast<uint32_t>(barrier_count))
            );
        }
    }
}

void RenderTargetManager::resize_swapchain(vk::Extent2D swapchain_extent) {
    this->swapchain_extent = swapchain_extent;
    this->targets.for_each([&](SlotKey<Target>, Target& target) {
        if (target.owned) {
            this->recreate_target(target);
        }
    });
}

static vk::Extent3D get_extent(const SizePolicy& policy, vk::Extent2D swapchain_extent) {
    if (auto adjusted = std::get_if<SwapchainAdjustedSizePolicy>(&policy); adjusted != nullptr) {
        auto width = std::round(static_cast<double>(swapchain_extent.width) * adjusted->scale);
        auto height = std::round(static_cast<double>(swapchain_extent.height) * adjusted->scale);
        return {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            1
        };
    } else if (auto fixed = std::get_if<FixedSizePolicy>(&policy); fixed != nullptr) {
        return vk::Extent3D(fixed->extent, 1);
    }
    return {};
}

vk::ImageUsageFlags get_required_usage(vk::Format format) {
    vk::ImageAspectFlags aspect = get_default_aspect_flags(format);
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled;
    if (aspect & (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)) {
        usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
    } else if (aspect & (vk::ImageAspectFlagBits::eColor)) {
        usage |= vk::ImageUsageFlagBits::eColorAttachment;
    }
    return usage;
}

bool RenderTargetManager::initialize_target(Target& target, const RenderTargetInfo& info) {
    if (target.key.has_value()) {
        this->bindless_set.free_texture(target.key.value());
        target.key = std::nullopt;
    }

    auto key = this->bindless_set.add_texture([&](const DeviceHandle& device) {
        return Texture(
            device,
            vk::ImageCreateInfo()
                .setImageType(vk::ImageType::e2D)
                .setUsage(info.usage | get_required_usage(info.format))
                .setFormat(info.format)
                .setExtent(get_extent(info.size_policy, this->swapchain_extent))
                .setMipLevels(info.mip_levels)
                .setArrayLayers(info.array_layers)
                .setSamples(info.samples)
                .setInitialLayout(vk::ImageLayout::eUndefined)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setTiling(vk::ImageTiling::eOptimal)
        );
    });

    if (!key.has_value()) {
        return false;
    }
    
    target.owned = true;
    target.key = key;
    target.array_layers = info.array_layers;
    target.mip_levels = info.mip_levels;
    target.size_policy = info.size_policy;
    target.states.clear();
    target.states.resize(static_cast<size_t>(target.array_layers) * target.mip_levels, {});
    return true;
}

std::optional<RenderTargetManager::Target> RenderTargetManager::create_target(
    const RenderTargetInfo& info
) {
    Target target;
    if (this->initialize_target(target, info)) {
        return target;
    }
    return std::nullopt;
}

bool RenderTargetManager::recreate_target(Target& target) {
    assert(target.key.has_value());
    const Texture* texture = this->get_target_texture(target);
    return this->initialize_target(
        target,
        RenderTargetInfo()
            .set_size_policy(target.size_policy.value())
            .set_array_layers(target.array_layers)
            .set_mip_levels(target.mip_levels)
            .set_format(texture->format())
            .set_samples(texture->samples())
            .set_usage(texture->usage())
    );
}

const Texture* RenderTargetManager::get_target_texture(const Target& target) const {
    if (!target.owned) {
        return target.bound;
    } else {
        return this->bindless_set.get_texture(target.key.value());
    }
}
