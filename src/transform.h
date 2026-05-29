#pragma once
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

struct Transform {
    glm::vec3 translation = glm::vec3(0.0f);
    glm::quat rotation = glm::identity<glm::quat>();
    float scale = 1.0f;

    Transform(glm::vec3 translation, glm::quat rotation, float scale): 
        translation(translation), rotation(rotation), scale(scale) {}

    Transform(glm::vec3 translation, glm::quat rotation): 
        translation(translation), rotation(rotation) {}

    Transform(glm::vec3 translation, float scale): 
        translation(translation), scale(scale) {}

    Transform(glm::vec3 translation): 
        translation(translation) {}

    Transform(float x, float y, float z): 
        translation(glm::vec3(x, y, z)) {}

    Transform(glm::quat rotation, float scale): 
        rotation(rotation), scale(scale) {}
    
    Transform(glm::quat rotation): 
        rotation(rotation) {}

    Transform(float scale): 
        scale(scale) {}
    
    Transform() = default;

    Transform& set_translation(glm::vec3 translation) {
        this->translation = translation;
        return *this;
    }

    Transform& set_rotation(glm::quat rotation) {
        this->rotation = rotation;
        return *this;
    }

    Transform& set_scale(float scale) {
        this->scale = scale;
        return *this;
    }

    Transform& look_at(glm::vec3 target, glm::vec3 up) {
        this->rotation = glm::quatLookAtLH(glm::normalize(target - this->translation), up);
        return *this;
    }

    glm::vec3 forward() const {
        return this->rotation * glm::vec3(0.0, 0.0, 1.0);
    }

    glm::vec3 back() const {
        return this->rotation * glm::vec3(0.0, 0.0, -1.0);
    }

    glm::vec3 down() const {
        return this->rotation * glm::vec3(0.0, 1.0, 0.0);
    }

    glm::vec3 up() const {
        return this->rotation * glm::vec3(0.0, -1.0, 0.0);
    }

    glm::vec3 right() const {
        return this->rotation * glm::vec3(1.0, 0.0, 0.0);
    }

    glm::vec3 left() const {
        return this->rotation * glm::vec3(-1.0, 0.0, 0.0);
    }
    
    glm::mat4 matrix() const {
        glm::mat4 matrix(glm::mat3_cast(this->rotation) * this->scale);
        matrix[3] = glm::vec4(this->translation, 1.0f);
        return matrix;
    }

    glm::mat4 inverse_matrix() const {
        float inv_scale = 1.0f / this->scale;
        glm::quat inv_rotation = glm::inverse(this->rotation);
        glm::mat4 matrix(glm::mat3_cast(inv_rotation) * inv_scale);
        return glm::translate(matrix, -this->translation);
    }

    glm::mat4x3 affine_matrix() const {
        glm::mat4x3 matrix(glm::mat3_cast(this->rotation) * this->scale);
        matrix[3] = this->translation;
        return matrix;
    }

    Transform operator*(const Transform& child) const {
        return Transform(
            this->translation + (this->rotation * (child.translation * this->scale)),
            this->rotation * child.rotation,
            this->scale * child.scale
        );
    }

    Transform& operator*=(const Transform& child) {
        *this = *this * child;
        return *this;
    }
};