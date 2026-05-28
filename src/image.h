#pragma once
#include <cstdint>
#include <cstddef>
#include <optional>
#include <vector>

struct Image {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t components = 0;
    std::vector<uint8_t> bytes;

    static std::optional<Image> load(const char *filename);

    Image& set_size(uint32_t width, uint32_t height) {
        this->width = width;
        this->height = height;
        return *this;
    }

    Image& set_components(uint32_t components) {
        this->components = components;
        return *this;
    }

    Image& set_bytes(std::vector<uint8_t>&& bytes) {
        this->bytes = bytes;
        return *this;
    }

    Image& set_bytes(const std::vector<uint8_t>& bytes) {
        this->bytes = bytes;
        return *this;
    }

    Image& set_bytes(const uint8_t* bytes, size_t count) {
        this->bytes = std::vector(bytes, bytes + count);
        return *this;
    }
};