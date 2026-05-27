#pragma once
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>

struct Camera {
    glm::vec3 translation = glm::vec3(0.0f);
    float pitch_radians = 0.0f;
    float yaw_radians = 0.0f;

    float aspect_ratio = 1.0f;
    float z_near = 0.001f;
    float fov_radians = glm::radians(90.0f);

    glm::vec3 forward() const {
        return glm::angleAxis(this->yaw_radians, glm::vec3(0.0, 1.0, 0.0)) *
            glm::angleAxis(this->pitch_radians, glm::vec3(1.0, 0.0, 0.0)) *
            glm::vec3(0.0, 0.0, 1.0);
    }

    glm::vec3 right() const {
        return glm::angleAxis(this->yaw_radians, glm::vec3(0.0, 1.0, 0.0)) *
            glm::angleAxis(this->pitch_radians, glm::vec3(1.0, 0.0, 0.0)) *
            glm::vec3(1.0, 0.0, 0.0);
    }

    glm::mat4 world_to_clip() const {
        return this->view_to_clip() * this->world_to_view();
    }

    glm::mat4 world_to_view() const {
        glm::quat rotation_quat = glm::angleAxis(-this->pitch_radians, glm::vec3(1.0, 0.0, 0.0)) *
            glm::angleAxis(-this->yaw_radians, glm::vec3(0.0, 1.0, 0.0));
        glm::mat4 rotation_mat = glm::mat4_cast(rotation_quat);
        return glm::translate(rotation_mat, -this->translation);
    }

    glm::mat4 view_to_clip() const {
        float f = 1.0f / std::tan(this->fov_radians * 0.5f);
        auto m = glm::mat4(0.0f);
        m[0][0] = f / this->aspect_ratio;
        m[1][1] = f;
        m[2][3] = 1.0f;
        m[3][2] = this->z_near;
        return m;
    }
};
