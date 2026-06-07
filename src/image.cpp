#define STB_IMAGE_IMPLEMENTATION
#include <spdlog/spdlog.h>
#include <stb_image.h>
#include "image.h"

size_t get_format_pixel_size_bytes(ImageFormat format) {
    switch (format) {
        case ImageFormat::R8:
        case ImageFormat::R8Srgb:
            return 1;
        case ImageFormat::Rg8:
        case ImageFormat::Rg8Srgb:
            return 2;
        case ImageFormat::Rgba8:
        case ImageFormat::Rgba8Srgb:
            return 4;
        case ImageFormat::Undefined:
            return 1;
    }
    return 0;
}

static ImageFormat get_image_format(int byte_components, ImageColorspace colorspace) {
    assert(byte_components > 0 && byte_components != 3 && byte_components <= 4);
    ImageFormat format = ImageFormat::Undefined;
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

    uint8_t* loaded_bytes = stbi_load(filename, &width, &height, nullptr, components);
    if (loaded_bytes == nullptr) {
        return std::nullopt;
    }

    size_t total_bytes = static_cast<size_t>(components)
        * static_cast<size_t>(width)
        * static_cast<size_t>(height);

    return Image()
        .set_size(static_cast<uint32_t>(width), static_cast<uint32_t>(height))
        .set_format(get_image_format(components, info.colorspace))
        .set_bytes(loaded_bytes, total_bytes);
}