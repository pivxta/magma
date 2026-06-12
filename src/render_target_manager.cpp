#include "render_target_manager.h"
#include <cmath>
#include <spdlog/spdlog.h>

RenderTargetManager::RenderTargetManager(
    const DeviceHandle& device, 
    uint32_t frames_in_flight,
    uint32_t max_targets
): 
    bindless_set(device, frames_in_flight, max_targets),
    targets(max_targets)
{
    this->frames_in_flight = frames_in_flight;
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

size_t RenderTargetManager::get_state_index(
    const Target& target, 
    uint32_t array_layer, 
    uint32_t mip_level
) {
    if (target.buffering == RenderTargetBuffering::PerFif && target.owned) {
        return this->frame_index * target.array_layers * target.mip_levels
            + array_layer * target.mip_levels
            + mip_level;
    } else {
        return array_layer * target.mip_levels + mip_level;
    }
}

static uint32_t get_texture_count(RenderTargetBuffering buffering, uint32_t frames_in_flight) {
    if (buffering == RenderTargetBuffering::PerFif) { 
        return frames_in_flight;
    } else {
        return 1;
    }
}

std::optional<RenderTargetId> RenderTargetManager::reserve() {
    if (auto key = this->targets.insert({}); key.has_value()) {
        return get_target_id(key.value());
    }
    return std::nullopt;
}

bool RenderTargetManager::bind(
    RenderTargetId id, 
    const Texture& texture, 
    RenderTargetSubresourceState state
) {
    assert(texture.mip_levels() == 1 && texture.array_layers() == 1);
    if (auto target = this->targets.get(get_slot_key(id)); target != nullptr) {
        if (target->owned) {
            return false;
        }
        target->textures.resize(1);
        target->textures[0] = &texture;
        target->states.resize(1);
        target->states[0] = state;
        target->mip_levels = texture.mip_levels();
        target->array_layers = texture.array_layers();
        return true;
    }
    return false;
}

std::optional<RenderTargetId> RenderTargetManager::add(const RenderTargetInfo& info) {
    if (auto key = this->targets.reserve(); key.has_value()) {
        Target target{
            .buffering = info.buffering,
            .size_policy = info.size_policy,
            .mip_levels = info.mip_levels,
            .array_layers = info.array_layers,
            .owned = true,
        };
        if (!this->create_textures(target, info)) {
            this->targets.free(key.value(), [](auto&){});
            return std::nullopt;
        }
        *this->targets.get(key.value()) = target;
        return get_target_id(key.value());
    }
    return std::nullopt;
}

std::optional<RenderTargetIndices> RenderTargetManager::get_indices(
    RenderTargetId id,
    Sampler sampler
) {
    if (auto target = this->targets.get(get_slot_key(id)); target != nullptr) {
        return RenderTargetIndices{
            .texture = target->keys[this->frame_index].index,
            .sampler = this->bindless_set.get_sampler(sampler)
        };
    }
    return std::nullopt;
}

const Texture* RenderTargetManager::get(RenderTargetId id) {
    if (auto target = this->targets.get(get_slot_key(id)); target != nullptr) {
        return this->get_target_texture(*target);
    }
    return nullptr;
}

void RenderTargetManager::free(RenderTargetId id) {
    this->targets.free(get_slot_key(id), [&](Target& target) {
        for (auto key: target.keys) {
            this->bindless_set.free_texture(key);
        }
        target.keys = {};
        target.textures = {};
        target.states = {};
    });
}

RenderTargetSubresourceState RenderTargetManager::state(
    RenderTargetId id, 
    uint32_t array_layer,
    uint32_t mip_level
) {
    if (auto target = this->targets.get(get_slot_key(id)); target != nullptr) {
        return target->states[this->get_state_index(*target, array_layer, mip_level)];
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
    if (auto target = this->targets.get(get_slot_key(id)); target != nullptr) {
        constexpr size_t MAX_BARRIERS = 128; 

        auto texture = this->get_target_texture(*target);
        auto aspect = get_default_aspect_flags(texture->format());
        size_t barrier_count = 0;
        std::array<vk::ImageMemoryBarrier2, MAX_BARRIERS> barriers;
        for (const auto& usage: usages) {
            for (uint32_t layer_offset = 0; layer_offset < usage.array_layer_count; layer_offset++) {
                for (uint32_t mip_offset = 0; mip_offset < usage.mip_level_count; mip_offset++) {
                    uint32_t array_layer = usage.base_array_layer + layer_offset;
                    uint32_t mip_level = usage.base_mip_level + mip_offset;
                    auto& state = target->states[this->get_state_index(
                        *target, 
                        array_layer, 
                        mip_level
                    )];

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
                                .setBaseArrayLayer(array_layer)
                                .setLayerCount(1)
                                .setBaseMipLevel(mip_level)
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
                    .setImageMemoryBarrierCount(barrier_count)
            );
        }
    }
}

void RenderTargetManager::resize_swapchain(vk::Extent2D swapchain_extent) {
    this->swapchain_extent = swapchain_extent;
    this->targets.for_each([&](SlotKey<Target>, Target& target) {
        if (target.owned) {
            this->recreate_textures(target);
        }
    });
}

void RenderTargetManager::begin_frame(uint64_t frame_counter) {
    this->bindless_set.begin_frame(frame_counter);
    this->frame_index = frame_counter % this->frames_in_flight;
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
    vk::ImageUsageFlags usage;
    if (aspect & (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)) {
        usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
    } else if (aspect & (vk::ImageAspectFlagBits::eColor)) {
        usage |= vk::ImageUsageFlagBits::eColorAttachment;
    }
    return usage;
}

bool RenderTargetManager::create_textures(Target& target, const RenderTargetInfo& info) {
    target.keys.clear();
    target.textures.clear();

    uint32_t texture_count = get_texture_count(info.buffering, this->frames_in_flight);
    for (uint32_t i = 0; i < texture_count; i++) {
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

        if (key.has_value()) {
            target.keys.push_back(key.value());
            target.textures.push_back(this->bindless_set.get_texture(key.value()));
        } else {
            for (auto created_key: target.keys) {
                this->bindless_set.free_texture(created_key);
            }
            target.keys.clear();
            target.textures.clear();
            return false;
        }
    }
    target.states.clear();
    target.states.resize(
        static_cast<size_t>(this->frames_in_flight)
            * static_cast<size_t>(target.array_layers) 
            * static_cast<size_t>(target.mip_levels),
        RenderTargetSubresourceState{}
    );
    return true;
}

bool RenderTargetManager::recreate_textures(Target& target) {
    assert(target.textures.size() >= 1);

    const auto info = RenderTargetInfo()
        .set_buffering(target.buffering)
        .set_size_policy(target.size_policy.value())
        .set_array_layers(target.array_layers)
        .set_mip_levels(target.mip_levels)
        .set_format(target.textures[0]->format())
        .set_samples(target.textures[0]->samples())
        .set_usage(target.textures[0]->usage());

    for (auto texture_key: target.keys) {
        this->bindless_set.free_texture(texture_key);
    }

    return this->create_textures(target, info);
}

const Texture* RenderTargetManager::get_target_texture(const Target& target) {
    if (target.buffering == RenderTargetBuffering::PerFif) {
        return target.textures[this->frame_index];
    } else {
        return target.textures[0];
    }
}