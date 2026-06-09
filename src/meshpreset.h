#pragma once
#include "mesh.h"
#include <glm/glm.hpp>

enum class ShadingMode: uint8_t {
    Smooth,
    Flat
};

struct Quad {
    glm::vec3 normal = glm::vec3(0.0f, -1.0f, 0.0f);
    float size = 1.0f;

    Quad() = default;
    Quad(float size) {
        this->size = size;
    }
    Quad(glm::vec3 normal, float size = 1.0f) {
        this->normal = normal;
        this->size = size;
    }

    Mesh to_mesh();
};

struct Cuboid {
    glm::vec3 size = glm::vec3(1.0f);

    Cuboid() = default;
    Cuboid(float size) {
        this->size = glm::vec3(size);
    }
    Cuboid(glm::vec3 size) {
        this->size = size;
    }

    Mesh to_mesh();
};

struct Sphere {
    ShadingMode shading = ShadingMode::Smooth;
    uint32_t stacks = 16;
    uint32_t slices = 16;
    float radius = 0.5f;

    Sphere() = default;
    Sphere(float radius) {
        this->radius = radius;
    }

    Sphere& set_slices(uint32_t slices) {
        this->slices = slices;
        return *this;
    }

    Sphere& set_stacks(uint32_t stacks) {
        this->stacks = stacks;
        return *this;
    }

    Sphere& set_radius(float radius) {
        this->radius = radius;
        return *this;
    }

    Sphere& set_shading(ShadingMode shading) {
        this->shading = shading;
        return *this;
    }

    Mesh to_mesh();
};

struct Cylinder {
    ShadingMode shading = ShadingMode::Smooth;
    uint32_t slices = 24;
    float radius = 0.5f;
    float height = 1.0f;

    Cylinder() = default;
    Cylinder(float height, float radius) {
        this->radius = radius;
        this->height = height;
    }

    Cylinder& set_height(float height) {
        this->height = height;
        return *this;
    }

    Cylinder& set_radius(float radius) {
        this->radius = radius;
        return *this;
    }

    Cylinder& set_slices(uint32_t slices) {
        this->slices = slices;
        return *this;
    }
    
    Cylinder& set_shading(ShadingMode shading) {
        this->shading = shading;
        return *this;
    }

    Mesh to_mesh();
};

