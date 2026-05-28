#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include "image.h"

std::optional<Image> Image::load(const char* filename) {
    int width, height, components;
    if (!stbi_info(filename, &width, &height, &components)) {
        return std::nullopt;
    }
    if (components == 3) {
        components = 4; // Most GPUs dont support Rgb8 format, so we use Rgba8.
    }

    uint8_t* loaded_bytes = stbi_load(filename, &width, &height, nullptr, components);
    if (loaded_bytes == nullptr) {
        return std::nullopt;
    }

    size_t total_bytes = static_cast<uint32_t>(components)
        * static_cast<uint32_t>(width)
        * static_cast<uint32_t>(height);

    return Image()
        .set_size(static_cast<uint32_t>(width), static_cast<uint32_t>(height))
        .set_components(static_cast<uint32_t>(components))
        .set_bytes(loaded_bytes, total_bytes);
}