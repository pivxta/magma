#include "texture_manager.h"
#include "image.h"
#include <cassert>
#include <cmath>

TextureManager::TextureManager(
    const DeviceHandle& device,
    Uploader& uploader,
    uint32_t frames_in_flight,
    uint32_t max_textures,
    const GlobalSamplerInfo& sampler_info
):
    bindless_set(device, frames_in_flight, max_textures, sampler_info)
{
    this->create_fallbacks(uploader);
}

static uint32_t get_fallback_index(TextureFallback fallback) {
    return static_cast<uint32_t>(fallback);
}

TextureId TextureManager::get_texture_id(SlotKey<Slot> key) {
    return {
        .index = key.index,
        .generation = key.generation
    };
}

TextureIndices TextureManager::get_slot_indices(const Slot& slot) {
    return {
        .texture = slot.texture.value_or(slot.fallback).index,
        .sampler = slot.sampler_index
    };
}

SlotKey<TextureManager::Slot> TextureManager::get_slot_key(TextureId id) {
    return {
        .index = id.index,
        .generation = id.generation
    };
}

TextureIndices TextureManager::get_fallback_indices(TextureFallback fallback) const {
    uint32_t fallback_index = get_fallback_index(fallback);
    uint32_t sampler_index = this->bindless_set.get_sampler(Sampler::NearestRepeat);
    uint32_t texture_index = this->fallbacks[fallback_index].index;
    return {
        .texture = texture_index,
        .sampler = sampler_index
    };
}

void TextureManager::update_pending() {
    this->bindless_set.update_pending();
}

TextureIndices TextureManager::get_fallback(TextureFallback fallback) const {
    return this->get_fallback_indices(fallback);
}

TextureIndices TextureManager::get(TextureId id, TextureFallback fallback) const {
    if (auto slot = this->slots.get(get_slot_key(id)); slot != nullptr) {
        return get_slot_indices(*slot);
    }
    return this->get_fallback_indices(fallback);
}

TextureId TextureManager::reserve(TextureFallback fallback) {
    Slot slot = {
        .sampler_index = this->bindless_set.get_sampler(Sampler::NearestRepeat),
        .fallback = this->fallbacks[get_fallback_index(fallback)]
    };
    return get_texture_id(this->slots.insert(slot).value());
}

TextureId TextureManager::add(
    Uploader& uploader,
    const Image& image,
    TextureFallback fallback
) {
    Slot slot = {
        .sampler_index = this->bindless_set.get_sampler(image.sampler),
        .texture = this->create_texture(uploader, image),
        .fallback = this->fallbacks[get_fallback_index(fallback)]
    };
    return get_texture_id(this->slots.insert(slot).value());
}

void TextureManager::set(
    TextureId id,
    Uploader& uploader,
    const Image& image,
    TextureFallback fallback
) {
    if (auto slot = this->slots.get(get_slot_key(id)); slot != nullptr) {
        if (slot->texture.has_value()) {
            this->bindless_set.free_texture(slot->texture.value());
        }
        Slot new_slot = {
            .sampler_index = this->bindless_set.get_sampler(image.sampler),
            .texture = this->create_texture(uploader, image),
            .fallback = this->fallbacks[get_fallback_index(fallback)]
        };
        *slot = new_slot;
        this->updated.push_back(id);
    }
}

void TextureManager::free(TextureId id) {
    this->slots.free(get_slot_key(id), [&](Slot& slot) {
        if (slot.texture.has_value()) {
            this->bindless_set.free_texture(slot.texture.value());
        }
        this->updated.push_back(id);
    });
}

static vk::Format get_image_format(ImageFormat image_format) {
    vk::Format format = vk::Format::eUndefined;
    switch (image_format) {
        case ImageFormat::R8Srgb: format = vk::Format::eR8Srgb; break;
        case ImageFormat::Rg8Srgb: format = vk::Format::eR8G8Srgb; break; 
        case ImageFormat::Rgba8Srgb: format = vk::Format::eR8G8B8A8Srgb; break;
        case ImageFormat::R8: format = vk::Format::eR8Unorm; break;
        case ImageFormat::Rg8: format = vk::Format::eR8G8Unorm; break; 
        case ImageFormat::Rgba8: format = vk::Format::eR8G8B8A8Unorm; break;
        case ImageFormat::Undefined: break;
    }
    return format;
}

static uint32_t get_mip_levels(uint32_t width, uint32_t height, std::optional<uint32_t> levels) {
    uint32_t max_mip_levels = static_cast<uint32_t>(std::log2(std::max(width, height))) + 1;
    return levels.value_or(max_mip_levels);
}

std::optional<SlotKey<Texture>> TextureManager::create_texture(
    Uploader& uploader,
    const Image& image
) {
    return this->bindless_set.add_texture([&](const DeviceHandle& device) {
        Texture texture(
            device,
            vk::ImageCreateInfo()
                .setImageType(vk::ImageType::e2D)
                .setExtent(vk::Extent3D(image.width, image.height, 1))
                .setFormat(get_image_format(image.format))
                .setMipLevels(get_mip_levels(image.width, image.height, image.mip_levels))
                .setArrayLayers(1)
                .setSamples(vk::SampleCountFlagBits::e1)
                .setUsage(vk::ImageUsageFlagBits::eTransferDst 
                    | vk::ImageUsageFlagBits::eTransferSrc
                    | vk::ImageUsageFlagBits::eSampled)
                .setInitialLayout(vk::ImageLayout::eUndefined)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setTiling(vk::ImageTiling::eOptimal)
        );
        this->mipmap_generator.generate(texture);
        uploader.upload_image(
            ImageUpload()
                .set_texture(texture)
                .set_image(image)
        );
        return texture;
    });
}

static Image create_color_fallback() {
    return Image()
        .set_size(2, 2)
        .set_format(ImageFormat::Rgba8Srgb)
        .set_sampler(Sampler::NearestRepeat)
        .set_bytes({
            255, 0, 255, 255,
            0,   0,   0, 255,
            0,   0,   0, 255,
            255, 0, 255, 255,
        });
}

static Image create_color_white_fallback() {
    return Image()
        .set_size(1, 1)
        .set_format(ImageFormat::Rgba8Srgb)
        .set_sampler(Sampler::NearestRepeat)
        .set_bytes({255, 255, 255, 255});
}

static Image create_normal_fallback() {
    return Image()
        .set_size(1, 1)
        .set_format(ImageFormat::Rgba8)
        .set_sampler(Sampler::NearestRepeat)
        .set_bytes({127, 127, 255, 255});
}

static Image create_displacement_fallback() {
    return Image()
        .set_size(1, 1)
        .set_format(ImageFormat::R8)
        .set_sampler(Sampler::NearestRepeat)
        .set_bytes({0});
}

static Image create_aorm_fallback() {
    return Image()
        .set_size(1, 1)
        .set_format(ImageFormat::Rgba8)
        .set_sampler(Sampler::NearestRepeat)
        .set_bytes({255, 255, 255, 255});
}

static Image create_fallback_image(TextureFallback type) {
    switch (type) {
        case TextureFallback::ColorWhite: return create_color_white_fallback();
        case TextureFallback::ColorError: return create_color_fallback();
        case TextureFallback::Normal: return create_normal_fallback();
        case TextureFallback::Displacement: return create_displacement_fallback();
        case TextureFallback::AoRoughnessMetallic: return create_aorm_fallback();
        case TextureFallback::Count: break;
    }
    return {};
}

void TextureManager::create_fallbacks(Uploader& uploader) {
    assert(this->bindless_set.texture_capacity() > static_cast<uint32_t>(TextureFallback::Count));
    std::array types = {
        TextureFallback::ColorWhite,
        TextureFallback::ColorError,
        TextureFallback::Normal,
        TextureFallback::Displacement,
        TextureFallback::AoRoughnessMetallic
    };
    for (auto type: types) {
        uint32_t index = get_fallback_index(type);
        Image image = create_fallback_image(type);
        this->fallbacks[index] = this->create_texture(uploader, image).value();
    }
}