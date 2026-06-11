#pragma once

struct TextureQualityInfo {
    float max_anisotropy = 1;
    float lod_bias = 0.0f;
    float min_lod = 0.0f;

    TextureQualityInfo& set_max_anisotroy(float anisotropy) {
        if (anisotropy > 1) {
            this->max_anisotropy = anisotropy;
        } else {
            this->max_anisotropy = 1;
        }
        return *this;
    }

    TextureQualityInfo& set_lod_bias(float lod_bias) {
        this->lod_bias = lod_bias;
        return *this;
    }

    TextureQualityInfo& set_min_lod(float min_lod) {
        this->min_lod = min_lod;
        return *this;
    }
};