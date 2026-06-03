#pragma once
#include <optional>
#include <glm/vec3.hpp>
#include "resource.h"
#include "textureindices.h"

struct Material {
    std::optional<TextureId> base_color_texture;
    std::optional<TextureId> normal_map;
    std::optional<TextureId> ao_roughness_metallic_map;
    glm::vec3 base_color_factor = glm::vec3(0.5f);
    float normal_factor = 1.0f;
    float roughness_factor = 0.5f;
    float metallic_factor = 0.0f;
    float ior = 1.5f;

    Material& set_base_color_texture(TextureId id) {
        this->base_color_texture = id;
        return *this;
    }

    Material& set_base_color_factor(glm::vec3 factor) {
        this->base_color_factor = factor;
        return *this;
    }

    Material& set_normal_map(TextureId id) {
        this->normal_map = id;
        return *this;
    }

    Material& set_normal_factor(float factor) {
        this->normal_factor = factor;
        return *this;
    }

    Material& set_ao_roughness_metallic_map(TextureId id) {
        this->metallic_factor = 1.0f;
        this->roughness_factor = 1.0f;
        this->ao_roughness_metallic_map = id;
        return *this;
    }

    Material& set_roughness_factor(float factor) {
        this->roughness_factor = factor;
        return *this;
    }

    Material& set_metallic_factor(float factor) {
        this->metallic_factor = factor;
        return *this;
    }

    Material& set_ior(float ior) {
        this->ior = ior;
        return *this;
    }
};