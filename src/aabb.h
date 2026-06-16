#pragma once
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <span>
#include <array>

struct Aabb {
    glm::vec3 min = glm::vec3(INFINITY);
    glm::vec3 max = glm::vec3(-INFINITY);

    Aabb() = default;

    explicit Aabb(glm::vec3 min, glm::vec3 max) {
        this->min = min;
        this->max = max;
    }
    
    explicit Aabb(std::span<const glm::vec3> points) {
        for (const auto point: points) {
            this->include(point);
        }
    }

    Aabb& set_min(glm::vec3 min) {
        this->min = min;
        return *this;
    }

    Aabb& set_max(glm::vec3 max) {
        this->max = max;
        return *this;
    }

    Aabb& include(glm::vec3 point) {
        this->min = glm::min(this->min, point);
        this->max = glm::max(this->max, point);
        return *this;
    }

    Aabb& merge(const Aabb& other) {
        this->min = glm::min(this->min, other.min);
        this->max = glm::max(this->max, other.max);
        return *this;
    }

    glm::vec3 center() const {
        return 0.5f * (this->min + this->max);
    }

    glm::vec3 extent() const {
        return this->max - this->min;
    }

    glm::vec3 half_extent() const {
        return 0.5f * this->extent();
    }

    std::array<glm::vec3, 8> corners() const {
        return {
            glm::vec3(this->min.x, this->min.y, this->min.z),
            glm::vec3(this->max.x, this->min.y, this->min.z),
            glm::vec3(this->min.x, this->max.y, this->min.z),
            glm::vec3(this->max.x, this->max.y, this->min.z),
            glm::vec3(this->min.x, this->min.y, this->max.z),
            glm::vec3(this->max.x, this->min.y, this->max.z),
            glm::vec3(this->min.x, this->max.y, this->max.z),
            glm::vec3(this->max.x, this->max.y, this->max.z),
        };
    }
};

static inline Aabb transform_aabb(
    const glm::mat3& linear, 
    const glm::vec3& translation, 
    const Aabb& box
) {
    const auto abs_linear = glm::mat3(
        glm::abs(linear[0]),
        glm::abs(linear[1]),
        glm::abs(linear[2])
    );
    const glm::vec3 center = linear * box.center() + translation;
    const glm::vec3 half_extent = abs_linear * box.half_extent();
    return Aabb(center - half_extent, center + half_extent);
}

static inline Aabb operator*(const glm::mat4& m, const Aabb& box) {
    return transform_aabb(glm::mat3(m), glm::vec3(m[3]), box);
}

static inline Aabb operator*(const glm::mat4x3& m, const Aabb& box) {
    return transform_aabb(glm::mat3(m), m[3], box);
}