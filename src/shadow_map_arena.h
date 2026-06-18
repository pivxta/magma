#pragma once
#include "render_target_manager.h"

struct CubeShadowMapIndices {
    RenderTargetIndices atlas;
    uint32_t cube;
};

struct ArrayShadowMapIndices {
    RenderTargetIndices atlas;
    uint32_t layer;
};

struct CubeShadowMap {
    RenderTargetId atlas;
    ComparisonSampler sampler;
    uint32_t base_layer;
    uint32_t layer_count;
    uint32_t resolution;

    CubeShadowMapIndices indices(const RenderTargetManager& render_targets) const;
};

struct ArrayShadowMap {
    RenderTargetId atlas;
    ComparisonSampler sampler;
    uint32_t base_layer;
    uint32_t layer_count;
    uint32_t resolution;

    ArrayShadowMapIndices indices(const RenderTargetManager& render_targets) const;
};

enum class ShadowMapFormat: uint8_t {
    Fixed16,
    Float32
};

class ArrayShadowMapArena {
public:
    ArrayShadowMapArena() = default;

    ArrayShadowMapArena(
        RenderTargetManager& render_targets,
        ShadowMapFormat format, 
        uint32_t resolution,
        ComparisonSampler sampler = ComparisonSampler::Linear
    );

    ArrayShadowMapArena(const ArrayShadowMapArena&) = delete;
    ArrayShadowMapArena& operator=(const ArrayShadowMapArena&) = delete;
    ArrayShadowMapArena(ArrayShadowMapArena&&) noexcept = default;
    ArrayShadowMapArena& operator=(ArrayShadowMapArena&&) noexcept = default;

    RenderTargetId atlas() const {
        return this->atlas_;
    }

    uint32_t resolution() const {
        return this->requested_resolution;
    }

    void set_resolution(uint32_t resolution) {
        this->requested_resolution = resolution;
    }

    void set_sampler(ComparisonSampler sampler) {
        this->sampler = sampler;
    }

    ArrayShadowMap allocate(uint32_t layer_count);

    void reset();
    void commit(RenderTargetManager& render_targets);

private:
    ComparisonSampler sampler;
    ShadowMapFormat format;
    RenderTargetId atlas_;
    uint32_t layers = 0;
    uint32_t current_resolution = 0;

    uint32_t used_layers = 0;
    uint32_t requested_resolution;
    bool in_allocation_stage = false;
};

class CubeShadowMapArena {
public:
    CubeShadowMapArena() = default;

    explicit CubeShadowMapArena(
        RenderTargetManager& render_targets,
        ShadowMapFormat format, 
        uint32_t resolution,
        ComparisonSampler sampler = ComparisonSampler::Linear
    ): arena(render_targets, format, resolution, sampler) {}

    CubeShadowMapArena(const CubeShadowMapArena&) = delete;
    CubeShadowMapArena& operator=(const CubeShadowMapArena&) = delete;
    CubeShadowMapArena(CubeShadowMapArena&&) noexcept = default;
    CubeShadowMapArena& operator=(CubeShadowMapArena&&) noexcept = default;

    RenderTargetId atlas() const {
        return this->arena.atlas();
    }

    uint32_t resolution() const {
        return this->arena.resolution();
    }

    void set_resolution(uint32_t resolution) {
        this->arena.set_resolution(resolution);
    }

    void set_sampler(ComparisonSampler sampler) {
        this->arena.set_sampler(sampler);
    }

    CubeShadowMap allocate() {
        ArrayShadowMap id = this->arena.allocate(6);
        return {
            .atlas = id.atlas,
            .sampler = id.sampler,
            .base_layer = id.base_layer,
            .layer_count = 6,
            .resolution = id.resolution
        };
    }

    void reset() {
        this->arena.reset();
    }

    void commit(RenderTargetManager& render_targets) {
        this->arena.commit(render_targets);
    }

private:
    ArrayShadowMapArena arena;
};

