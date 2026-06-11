#pragma once
#include <variant>
#include <glm/vec3.hpp>

struct AgxLook {
    glm::vec3 slope = glm::vec3(1.0f);
    glm::vec3 offset = glm::vec3(0.0f);
    glm::vec3 power = glm::vec3(1.0f);
    float saturation = 1.0f;

    AgxLook& set_slope(glm::vec3 slope) {
        this->slope = slope;
        return *this;
    }

    AgxLook& set_offset(glm::vec3 offset) {
        this->offset = offset;
        return *this;
    }

    AgxLook& set_power(glm::vec3 power) {
        this->power = power;
        return *this;
    }

    AgxLook& set_saturation(float saturation) {
        this->saturation = saturation;
        return *this;
    }

    static AgxLook basic() {
        return {};
    }
    
    static AgxLook golden() {
        return AgxLook()
            .set_saturation(0.8f)
            .set_power(glm::vec3(0.8f))
            .set_slope(glm::vec3(1.0f, 0.9f, 0.5f));
    }

    static AgxLook punchy() {
        return AgxLook()
            .set_saturation(1.4f)
            .set_power(glm::vec3(1.35f));
    }

    bool operator==(const AgxLook&) const = default;
};

struct AgxTonemap {
    AgxLook look;
    float exposure_factor = 1.0f;

    AgxTonemap& set_look(AgxLook look) {
        this->look = look;
        return *this;
    }

    AgxTonemap& set_exposure_factor(float factor) {
        this->exposure_factor = factor;
        return *this;
    }

    bool operator==(const AgxTonemap&) const = default;
};

struct ReinhardTonemap {
    float exposure_factor = 1.0f;

    ReinhardTonemap& set_exposure_factor(float factor) {
        this->exposure_factor = factor;
        return *this;
    }

    bool operator==(const ReinhardTonemap&) const = default;
};

struct DisabledTonemap {
    bool operator==(const DisabledTonemap&) const {
        return true;
    }
};

using Tonemap = std::variant<
    DisabledTonemap,
    ReinhardTonemap,
    AgxTonemap
>;

struct PostProcessing {
    Tonemap tonemap = AgxTonemap();
};