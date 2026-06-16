#pragma once
#include <variant>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include "transform.h"

struct OrthographicProjection {
    float viewport_height = 2.0f;
    float scale = 1.0f;
    float near = 0.001f;
    float far = 1000.0f;
    
    [[nodiscard]]
    glm::mat4 view_to_clip(float viewport_width, float viewport_height) const {
        float half_height = 0.5f * this->viewport_height * this->scale;
        float half_width = half_height * (viewport_width / viewport_height);
        return glm::orthoLH_ZO(
            -half_width, half_width,
            -half_height, half_height,
            this->far, this->near
        );
    }
};

struct PerspectiveProjection {
    float fov_radians = glm::radians(90.0f);
    float near = 0.001f;
    float far = 1000.0f;
    
    [[nodiscard]]
    glm::mat4 view_to_clip(float viewport_width, float viewport_height) const {
        return glm::perspectiveFovLH_ZO(
            this->fov_radians,
            viewport_width,
            viewport_height,
            this->far,
            this->near
        );
    }
};

using Projection = std::variant<
    PerspectiveProjection, 
    OrthographicProjection
>;

[[nodiscard]]
static inline glm::mat4 get_view_to_clip(const Projection& projection, float width, float height) {
    return std::visit([width, height](const auto& proj) -> glm::mat4 {
        return proj.view_to_clip(width, height);
    }, projection);
}

[[nodiscard]]
static inline float get_projection_near(const Projection& projection) {
    return std::visit([](const auto& proj) { return proj.near; }, projection);
}

[[nodiscard]]
static inline float get_projection_far(const Projection& projection) {
    return std::visit([](const auto& proj) { return proj.far; }, projection);
}

struct Film {
    float exposure = 1.0f;

    Film() = default;
    Film(float exposure) {
        this->exposure = exposure;
    }

    Film& set_exposure(float exposure) {
        this->exposure = exposure;
        return *this;
    }
};

struct Camera {
    Transform transform;
    Projection projection = PerspectiveProjection();
    Film film;
    bool active = true;

    Camera& set_transform(const Transform& transform) {
        this->transform = transform;
        return *this;
    }

    Camera& set_projection(const Projection& projection) {
        this->projection = projection;
        return *this;
    }

    Camera& set_active(bool active = true) {
        this->active = active;
        return *this;
    }

    [[nodiscard]]
    float near() const {
        return get_projection_near(this->projection);
    }

    [[nodiscard]]
    float far() const {
        return get_projection_far(this->projection);
    }

    [[nodiscard]]
    glm::mat4 world_to_view() const {
        return this->transform.inverse_matrix();
    }

    [[nodiscard]]
    glm::mat4 view_to_clip(float viewport_width, float viewport_height) const {
        return get_view_to_clip(this->projection, viewport_width, viewport_height);
    }

    [[nodiscard]]
    glm::mat4 world_to_clip(float viewport_width, float viewport_height) const {
        return this->view_to_clip(viewport_width, viewport_height) * this->world_to_view();
    }
};
