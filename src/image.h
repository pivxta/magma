#pragma once
#include <cstdint>
#include <cstddef>
#include <optional>
#include <vector>
#include <span>

enum class Sampler: uint8_t {
    LinearRepeat = 0,
    NearestRepeat,
    LinearMirrored,
    NearestMirrored,
    LinearClamp,
    NearestClamp,
    Count,
};

enum class ImageFormat: uint8_t {
    Undefined,
    R8,
    Rg8,
    Rgba8,
    R8Srgb,
    Rg8Srgb,
    Rgba8Srgb,
};

enum class ImageColorspace: uint8_t {
    Linear,
    Srgb
};

struct ImageLoadInfo {
    ImageColorspace colorspace = ImageColorspace::Srgb;

    ImageLoadInfo& set_colorspace(ImageColorspace colorspace) {
        this->colorspace = colorspace;
        return *this;
    }
};

size_t get_format_pixel_size_bytes(ImageFormat format);

struct Image {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> bytes;
    ImageFormat format = ImageFormat::Undefined;
    Sampler sampler = Sampler::LinearRepeat;
    std::optional<uint32_t> mip_levels;

    static std::optional<Image> load(
        const char *filename, 
        const ImageLoadInfo& info = {}
    );

    Image& set_size(uint32_t width, uint32_t height) {
        this->width = width;
        this->height = height;
        return *this;
    }

    Image& set_format(ImageFormat format) {
        this->format = format;
        return *this;
    }

    Image& set_bytes(std::vector<uint8_t>&& bytes) {
        this->bytes = bytes;
        return *this;
    }

    Image& set_bytes(std::span<const uint8_t> bytes) {
        this->bytes = std::vector(bytes.begin(), bytes.end());
        return *this;
    }

    Image& set_bytes(const uint8_t* bytes, size_t count) {
        this->bytes = std::vector(bytes, bytes + count);
        return *this;
    }

    Image& set_sampler(Sampler sampler) {
        this->sampler = sampler;
        return *this;
    }

    Image& set_mip_levels(uint32_t mip_levels) {
        this->mip_levels = mip_levels;
        return *this;
    }

    Image& set_auto_mip_levels() {
        this->mip_levels = std::nullopt;
        return *this;
    }

    size_t expected_size_bytes() const {
        return static_cast<size_t>(this->width) 
            * static_cast<size_t>(this->height)
            * get_format_pixel_size_bytes(this->format);
    }
};