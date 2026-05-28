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
    
    glm::mat4 view_to_clip(float viewport_width, float viewport_height) const {
        float aspect_ratio = viewport_width / viewport_height;
        float h = this->viewport_height * this->scale;
        float w = h * aspect_ratio;

        auto m = glm::mat4(0.0f);
        m[0][0] = 2.0f / w;
        m[1][1] = 2.0f / h;
        m[2][2] = 1.0f / (this->near - this->far);
        m[3][2] = -this->far / (this->near - this->far);
        m[3][3] = 1.0f;
        return m;
    }
};

struct PerspectiveProjection {
    float fov_radians = glm::radians(90.0f);
    float near = 0.001f;
    
    glm::mat4 view_to_clip(float viewport_width, float viewport_height) const {
        float f = 1.0f / std::tan(this->fov_radians * 0.5f);
        auto m = glm::mat4(0.0f);
        m[0][0] = f * viewport_height / viewport_width;
        m[1][1] = f;
        m[2][3] = 1.0f;
        m[3][2] = this->near;
        return m;
    }
};

using Projection = std::variant<
    PerspectiveProjection, 
    OrthographicProjection
>;

static glm::mat4 get_view_to_clip(const Projection& projection, float width, float height) {
    return std::visit([width, height](const auto& proj) -> glm::mat4 {
        return proj.view_to_clip(width, height);
    }, projection);
}

struct Camera {
    Transform transform;
    Projection projection = PerspectiveProjection();

    glm::mat4 world_to_view() const {
        return this->transform.inverse_matrix();
    }

    glm::mat4 view_to_clip(float viewport_width, float viewport_height) const {
        return get_view_to_clip(this->projection, viewport_width, viewport_height);
    }

    glm::mat4 world_to_clip(float viewport_width, float viewport_height) const {
        return this->view_to_clip(viewport_width, viewport_height) * this->world_to_view();
    }
};
