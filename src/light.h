#pragma once
#include <glm/vec3.hpp>

struct AmbientLight {
    glm::vec3 color;
    float illuminance;

    AmbientLight& set_color(glm::vec3 color) {
        this->color = color;
        return *this;
    }

    AmbientLight& set_illuminance(float illuminance) {
        this->illuminance = illuminance;
        return *this;
    }
};

struct DirectionalLight {
    glm::vec3 direction = glm::vec3(0.0, 1.0, 0.0);
    glm::vec3 color = glm::vec3(1.0);
    float illuminance = 10.0;

    bool shadows_enabled = false;
    float shadow_bias_constant_factor = 1.5f;
    float shadow_bias_slope_factor = 2.0f;

    DirectionalLight& set_direction(glm::vec3 direction) {
        this->direction = direction;
        return *this;
    }

    DirectionalLight& set_color(glm::vec3 color) {
        this->color = color;
        return *this;
    }

    DirectionalLight& set_illuminance(float illuminance) {
        this->illuminance = illuminance;
        return *this;
    }

    DirectionalLight& set_shadows_enabled(bool enabled) {
        this->shadows_enabled = enabled;
        return *this;
    }

    DirectionalLight& set_shadow_bias_constant_factor(float value) {
        this->shadow_bias_constant_factor = value;
        return *this;
    }

    DirectionalLight& set_shadow_bias_slope_factor(float value) {
        this->shadow_bias_slope_factor = value;
        return *this;
    }
};

struct PointLight {
    glm::vec3 position = glm::vec3(0.0, 0.0, 0.0);
    float radius = 30.0f;
    glm::vec3 color = glm::vec3(1.0);
    float intensity = 200.0;

    bool shadows_enabled = false;
    float shadow_bias_constant_factor = 1.5f;
    float shadow_bias_slope_factor = 2.0f;

    PointLight& set_position(glm::vec3 position) {
        this->position = position;
        return *this;
    }

    PointLight& set_color(glm::vec3 color) {
        this->color = color;
        return *this;
    }

    PointLight& set_intensity(float intensity) {
        this->intensity = intensity;
        return *this;
    }

    PointLight& set_radius(float radius) {
        this->radius = radius;
        return *this;
    }
};