#pragma once
#include <cstddef>
#include <array>
#include <glm/glm.hpp>
#include "light.h"
#include "camera.h"

static constexpr size_t MAX_CASCADES = 4;

struct CascadeInfo {
    uint32_t count = MAX_CASCADES;
    float lambda = 0.8f;
    float min_distance = 0.01f;
    float max_distance = 50.0f;
    uint32_t resolution;

    CascadeInfo(uint32_t resolution) {
        this->resolution = resolution;
    }

    CascadeInfo& set_lambda(float lambda) {
        this->lambda = lambda;
        return *this;
    }

    CascadeInfo& set_min_distance(float value) {
        this->min_distance = value;
        return *this;
    }

    CascadeInfo& set_max_distance(float value) {
        this->max_distance = value;
        return *this;
    }

    CascadeInfo& set_count(uint32_t count) {
        this->count = std::min(count, static_cast<uint32_t>(MAX_CASCADES));
        return *this;
    }
};

struct Cascades {
    uint32_t count = 0;
    std::array<float, MAX_CASCADES> depths = {};
    std::array<glm::mat4, MAX_CASCADES> world_to_clip = {};

    Cascades() = default;

    explicit Cascades(
        const CascadeInfo& info,
        const Camera& camera, 
        glm::vec2 viewport,
        const DirectionalLight& light
    );
};