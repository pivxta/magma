#pragma once
#include <cstdint>

enum class Sampler: uint8_t {
    LinearRepeat = 0,
    NearestRepeat,
    LinearMirrored,
    NearestMirrored,
    LinearClamp,
    NearestClamp,
    Count,
};

enum class Filter: uint8_t {
    Nearest,
    Linear
};

struct GlobalSamplerInfo {
    Filter minify_filter = Filter::Linear;
    Filter mip_map_filter = Filter::Linear;
    float max_anisotropy = 1;
    float lod_bias = 0.0f;
    float min_lod = 0.0f;

    GlobalSamplerInfo& set_minify_filter(Filter filter) {
        this->minify_filter = filter;
        return *this;
    }

    GlobalSamplerInfo& set_max_anisotroy(float anisotropy) {
        if (anisotropy > 1) {
            this->max_anisotropy = anisotropy;
        } else {
            this->max_anisotropy = 1;
        }
        return *this;
    }

    GlobalSamplerInfo& set_lod_bias(float lod_bias) {
        this->lod_bias = lod_bias;
        return *this;
    }

    GlobalSamplerInfo& set_min_lod(float min_lod) {
        this->min_lod = min_lod;
        return *this;
    }
};