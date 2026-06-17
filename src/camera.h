#pragma once
#include <variant>
#include <cmath>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <spdlog/spdlog.h>
#include "transform.h"

struct OrthographicProjection {
    float viewport_height = 2.0f;
    float scale = 1.0f;
    float near = 0.001f;
    float far = 1000.0f;

    [[nodiscard]]
    glm::mat4 clip_to_view_for_range(glm::vec2 viewport, float near, float far) const {
        const float half_height = 0.5f * this->viewport_height * this->scale;
        const float half_width = half_height * (viewport.x / viewport.y);
        return glm::inverse(glm::orthoLH_ZO(
            -half_width, half_width,
            -half_height, half_height,
            std::min(far, this->far), 
            std::max(near, this->near)
        ));
    }

    [[nodiscard]]
    glm::mat4 view_to_clip(glm::vec2 viewport) const {
        const float half_height = 0.5f * this->viewport_height * this->scale;
        const float half_width = half_height * (viewport.x / viewport.y);
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
    glm::mat4 clip_to_view_for_range(glm::vec2 viewport, float near, float far) const {
        return glm::inverse(
            glm::perspectiveFovLH_ZO(
                this->fov_radians, 
                viewport.x, viewport.y, 
                std::min(far, this->far), 
                std::max(near, this->near)
            )
        );
    }

    [[nodiscard]]
    glm::mat4 view_to_clip(glm::vec2 viewport) const {
        const float h = 1.0f / std::tan(0.5f * this->fov_radians);
        const float w = h * viewport.y / viewport.x;
        auto result = glm::zero<glm::mat4>();
        result[0][0] = w;
        result[1][1] = h;
        result[2][2] = 0.0f;
        result[2][3] = 1.0f;
        result[3][2] = this->near;
        return result;
    }

};

using Projection = std::variant<
    PerspectiveProjection, 
    OrthographicProjection
>;

[[nodiscard]]
static inline glm::mat4 get_projection_view_to_clip(const Projection& projection, glm::vec2 viewport) {
    return std::visit([=](const auto& proj) {
        return proj.view_to_clip(viewport);
    }, projection);
}

[[nodiscard]]
static inline std::array<glm::vec3, 8> get_projection_frustum_corners(
    const Projection& projection,
    glm::vec2 viewport, 
    float near, 
    float far
) {
    return std::visit([=](const auto& proj) {
        const auto clip_to_view = proj.clip_to_view_for_range(viewport, near, far);

        std::array<glm::vec3, 8> corners;
        for (int i = 0; i < 8; i++) {
            const auto corner = clip_to_view * glm::vec4(
                static_cast<float>(2 * (i % 2) - 1),
                static_cast<float>(2 * ((i / 2) % 2) - 1),
                static_cast<float>(1 - (i / 4) % 2),
                1.0f
            );
            corners[i] = corner / corner.w;
        }

        return corners;   
    }, projection);
}

[[nodiscard]]
static inline float get_projection_near(const Projection& projection) {
    return std::visit([=](const auto& proj) {
        return proj.near;
    }, projection);
}

[[nodiscard]]
static inline float get_projection_far(const Projection& projection) {
    return std::visit([=](const auto& proj) {
        return proj.far;
    }, projection);
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
    std::array<glm::vec3, 8> frustum_corners(glm::vec2 viewport, float near, float far) const {
        auto corners = get_projection_frustum_corners(this->projection, viewport, near, far);
        for (auto& corner: corners) {
            corner = this->transform * corner;
        }
        return corners;
    }

    [[nodiscard]]
    glm::mat4 world_to_view() const {
        return this->transform.inverse_matrix();
    }

    [[nodiscard]]
    glm::mat4 view_to_clip(glm::vec2 viewport) const {
        return get_projection_view_to_clip(this->projection, viewport);
    }

    [[nodiscard]]
    glm::mat4 world_to_clip(glm::vec2 viewport) const {
        return this->view_to_clip(viewport) * this->world_to_view();
    }
};
