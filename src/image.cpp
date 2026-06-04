#define STB_IMAGE_IMPLEMENTATION
#include <spdlog/spdlog.h>
#include <stb_image.h>
#include "image.h"

static ImageFormat get_image_format(int byte_components, ImageColorspace colorspace) {
    assert(byte_components > 0 && byte_components != 3 && byte_components <= 4);
    ImageFormat format;
    if (colorspace == ImageColorspace::Srgb) {
        switch (byte_components) {
            case 1: format = ImageFormat::R8Srgb; break;
            case 2: format = ImageFormat::Rg8Srgb; break;
            case 4: format = ImageFormat::Rgba8Srgb; break;
            default: break;
        }
    } else {
        switch (byte_components) {
            case 1: format = ImageFormat::R8; break;
            case 2: format = ImageFormat::Rg8; break;
            case 4: format = ImageFormat::Rgba8; break;
            default: break;
        }
    }
    return format; 
}

std::optional<Image> Image::load(const char* filename, const ImageLoadInfo& info) {
    int width, height, components;
    if (!stbi_info(filename, &width, &height, &components)) {
        return std::nullopt;
    }
    if (components == 3) {
        components = 4; // Most GPUs dont support Rgb8 format, so we use Rgba8.
    }
    spdlog::info("{}: {} channels", filename, components);

    uint8_t* loaded_bytes = stbi_load(filename, &width, &height, nullptr, components);
    if (loaded_bytes == nullptr) {
        return std::nullopt;
    }

    size_t total_bytes = static_cast<uint32_t>(components)
        * static_cast<uint32_t>(width)
        * static_cast<uint32_t>(height);

    return Image()
        .set_size(static_cast<uint32_t>(width), static_cast<uint32_t>(height))
        .set_format(get_image_format(components, info.colorspace))
        .set_bytes(loaded_bytes, total_bytes);
}