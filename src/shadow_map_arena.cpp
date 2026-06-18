#include "shadow_map_arena.h"

CubeShadowMapIndices CubeShadowMap::indices(const RenderTargetManager& render_targets) const {
    const auto atlas = render_targets.get_indices(this->atlas, this->sampler);
    assert(atlas.has_value() && "Invalid shadow map ID");
    assert(base_layer % 6 == 0 && "Invalid shadow cubemap array range");
    return {
        .atlas = atlas.value(),
        .cube = this->base_layer / 6
    };
}

ArrayShadowMapIndices ArrayShadowMap::indices(const RenderTargetManager& render_targets) const {
    const auto atlas = render_targets.get_indices(this->atlas, this->sampler);
    assert(atlas.has_value() && "Invalid shadow map ID");
    return {
        .atlas = atlas.value(),
        .layer = this->base_layer
    };
}

ArrayShadowMapArena::ArrayShadowMapArena(
    RenderTargetManager& render_targets,
    ShadowMapFormat format, 
    uint32_t resolution,
    ComparisonSampler sampler
) {
    this->format = format;
    this->sampler = sampler;
    this->requested_resolution = resolution;
    this->atlas_ = render_targets.reserve().value();
}

ArrayShadowMap ArrayShadowMapArena::allocate(uint32_t layer_count) {
    assert(this->in_allocation_stage);
    uint32_t base_layer = this->used_layers;
    this->used_layers += layer_count;
    return {
        .atlas = this->atlas_,
        .sampler = this->sampler,
        .base_layer = base_layer,
        .layer_count = layer_count,
        .resolution = this->requested_resolution
    };
}

void ArrayShadowMapArena::reset() {
    this->used_layers = 0;
    this->in_allocation_stage = true;
}

static inline vk::Format get_format(ShadowMapFormat format) {
    switch (format) {
        case ShadowMapFormat::Fixed16: return vk::Format::eD16Unorm;
        case ShadowMapFormat::Float32: return vk::Format::eD32Sfloat;
    }
    return vk::Format::eUndefined;
}

static inline uint32_t next_power_of_two(uint32_t n) {
    if (n == 0) {
        return 1;
    }

    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

void ArrayShadowMapArena::commit(RenderTargetManager& render_targets) {
    this->in_allocation_stage = false;
    if (this->requested_resolution != this->current_resolution 
        || this->used_layers > this->layers) 
    {
        this->layers = next_power_of_two(this->used_layers);
        this->current_resolution = this->requested_resolution;
        render_targets.reset(
            this->atlas_,
            RenderTargetInfo()
                .set_format(get_format(this->format))
                .set_array_layers(this->layers)
                .set_size_policy(FixedSizePolicy(
                    this->current_resolution, 
                    this->current_resolution
                ))
        );
    }
}