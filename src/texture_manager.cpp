#include "texture_manager.h"
#include "image.h"
#include "vk_error.h"
#include <cassert>
#include <spdlog/spdlog.h>

static constexpr uint32_t FALLBACK_COUNT = static_cast<uint32_t>(TextureFallback::Count);
static constexpr uint32_t SAMPLER_COUNT = static_cast<uint32_t>(Sampler::Count);

static vk::DescriptorPool create_descriptor_pool(vk::Device device, uint32_t max_images) {
    std::array pool_sizes = {
        vk::DescriptorPoolSize()
            .setType(vk::DescriptorType::eSampler)
            .setDescriptorCount(SAMPLER_COUNT),
        vk::DescriptorPoolSize()
            .setType(vk::DescriptorType::eSampledImage)
            .setDescriptorCount(max_images),
    };
    auto [result, descriptor_pool] = device.createDescriptorPool(
        vk::DescriptorPoolCreateInfo()
            .setMaxSets(1)
            .setPoolSizes(pool_sizes)
            .setFlags(vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind)
    );
    vk_expect(result, "Failed to create descriptor pool");
    return descriptor_pool;
}

static vk::DescriptorSetLayout create_set_layout(vk::Device device, uint32_t max_images) {
    std::array bindings = {
        vk::DescriptorSetLayoutBinding()
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eSampledImage)
            .setDescriptorCount(max_images)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment),
        vk::DescriptorSetLayoutBinding()
            .setBinding(1)
            .setDescriptorType(vk::DescriptorType::eSampler)
            .setDescriptorCount(SAMPLER_COUNT)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment),
    };

    std::array<vk::DescriptorBindingFlags, 2> binding_flags = {
        vk::DescriptorBindingFlagBits::eUpdateAfterBind
            | vk::DescriptorBindingFlagBits::ePartiallyBound,
        vk::DescriptorBindingFlagBits::ePartiallyBound, 
    };

    auto [result1, layout] = device.createDescriptorSetLayout(
        vk::StructureChain {
            vk::DescriptorSetLayoutCreateInfo()
                .setBindings(bindings)
                .setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool),
            vk::DescriptorSetLayoutBindingFlagsCreateInfo()
                .setBindingFlags(binding_flags)
        }.get()
    );
    vk_expect(result1, "Failed to create descriptor set layout");
    return layout;
}

static vk::DescriptorSet create_set(
    vk::Device device,
    vk::DescriptorPool pool,
    vk::DescriptorSetLayout layout
) {
    auto [result2, sets] = device.allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo()
            .setDescriptorPool(pool)
            .setSetLayouts(layout)
    );
    vk_expect(result2, "Failed to create descriptor set layout");
    return sets[0];
}

TextureManager::TextureManager(
    vk::Device device, 
    vma::Allocator allocator,
    Uploader& uploader,
    uint32_t frames_in_flight,
    uint32_t max_textures
):
    device(device),
    allocator(allocator),
    textures(max_textures),
    frames_in_flight(frames_in_flight)
{
    assert(max_textures > FALLBACK_COUNT);
    this->desc_pool = create_descriptor_pool(device, max_textures);
    this->desc_set_layout = create_set_layout(device, max_textures);
    this->desc_set = create_set(device, this->desc_pool, this->desc_set_layout);
    this->create_samplers(device);
    this->create_fallbacks(uploader);
}

void TextureManager::destroy() {
    this->device.destroyDescriptorSetLayout(this->desc_set_layout);
    this->device.destroyDescriptorPool(this->desc_pool);
    this->destroy_samplers(this->device);
    this->textures.clear([&](Texture& texture) {
        texture.destroy(this->device, this->allocator);
    });
}

static uint32_t get_sampler_index(Sampler sampler) {
    return static_cast<uint32_t>(sampler);
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
    uint32_t sampler_index = get_sampler_index(Sampler::NearestRepeat);
    uint32_t fallback_index = get_fallback_index(fallback);
    uint32_t texture_index = this->fallbacks[fallback_index].index;
    return {
        .texture = texture_index,
        .sampler = sampler_index
    };
}

void TextureManager::destroy_pending() {
    while (!this->destroy_queue.empty()) {
        const PendingDestroy& pending = this->destroy_queue.back();
        if (this->frame_counter - pending.request_frame >= this->frames_in_flight) {
            this->destroy_texture(pending.texture);
            this->destroy_queue.pop_back();
        } else {
            break;
        }
    }
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
        .sampler_index = get_sampler_index(Sampler::NearestRepeat),
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
        .sampler_index = get_sampler_index(image.sampler),
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
            this->destroy_queue.push_front(PendingDestroy{
                .request_frame = this->frame_counter,
                .texture = *slot->texture
            });
        }
        Slot new_slot = {
            .sampler_index = get_sampler_index(image.sampler),
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
            this->destroy_queue.push_front(PendingDestroy{
                .request_frame = this->frame_counter,
                .texture = slot.texture.value()
            });
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

static Texture create_texture_from_image(
    vk::Device device, 
    vma::Allocator allocator,
    const Image& src
) {
    vk::Extent3D extent = vk::Extent3D(src.width, src.height, 1);
    vk::Format format = get_image_format(src.format);
    uint32_t max_mip_levels = static_cast<uint32_t>(std::log2(std::max(src.width, src.height))) + 1;
    uint32_t mip_levels = src.mip_levels.value_or(max_mip_levels);

    return create_texture(
        device,
        allocator,
        vk::ImageCreateInfo()
            .setImageType(vk::ImageType::e2D)
            .setExtent(extent)
            .setFormat(format)
            .setMipLevels(mip_levels)
            .setArrayLayers(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setUsage(vk::ImageUsageFlagBits::eTransferDst 
                | vk::ImageUsageFlagBits::eTransferSrc
                | vk::ImageUsageFlagBits::eSampled)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setTiling(vk::ImageTiling::eOptimal)
    );
}

std::optional<SlotKey<Texture>> TextureManager::create_texture(
    Uploader& uploader,
    const Image& image
) {
    if (auto key = this->textures.reserve(); key.has_value()) {
        Texture texture = create_texture_from_image(
            this->device,
            this->allocator,
            image
        );
        uploader.upload_image(
            ImageUpload()
                .set_texture(texture)
                .set_image(image)
        );
        this->mipmap_generator.generate(texture);

        auto info = vk::DescriptorImageInfo()
            .setImageView(texture.view)
            .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
        device.updateDescriptorSets(
            vk::WriteDescriptorSet()
                .setDstSet(this->desc_set)
                .setDstBinding(0)
                .setDstArrayElement(key->index)
                .setDescriptorType(vk::DescriptorType::eSampledImage)
                .setImageInfo(info),
            {}
        );
        *this->textures.get(*key) = texture;
        return key;
    }
    return std::nullopt;
}

void TextureManager::destroy_texture(SlotKey<Texture> key) {
    this->textures.free(key, [&](Texture& texture) {
        texture.destroy(this->device, this->allocator);
        texture = Texture{};
    });
}

static vk::SamplerAddressMode get_address_mode(Sampler sampler) {
    switch (sampler) {
        case Sampler::LinearRepeat: return vk::SamplerAddressMode::eRepeat;
        case Sampler::LinearMirrored: return vk::SamplerAddressMode::eMirroredRepeat;
        case Sampler::LinearClamp: return vk::SamplerAddressMode::eClampToEdge;
        case Sampler::NearestRepeat: return vk::SamplerAddressMode::eRepeat;
        case Sampler::NearestMirrored: return vk::SamplerAddressMode::eMirroredRepeat;
        case Sampler::NearestClamp: return vk::SamplerAddressMode::eClampToEdge;
        default: return vk::SamplerAddressMode::eRepeat;
    }
}

static vk::Filter get_filter(Sampler sampler) {
    switch (sampler) {
        case Sampler::LinearRepeat:
        case Sampler::LinearMirrored:
        case Sampler::LinearClamp: 
            return vk::Filter::eLinear;
        case Sampler::NearestRepeat: 
        case Sampler::NearestMirrored:
        case Sampler::NearestClamp: 
        default:
            return vk::Filter::eNearest;
    }
}

static vk::Sampler create_sampler(vk::Device device, Sampler type) {
    vk::SamplerAddressMode address_mode = get_address_mode(type);
    vk::Filter filter = get_filter(type);
    auto info = vk::SamplerCreateInfo()
        .setMagFilter(filter)
        .setAddressModeU(address_mode)
        .setAddressModeV(address_mode)
        .setAddressModeW(address_mode)
        .setMinFilter(vk::Filter::eLinear)
        .setMipmapMode(vk::SamplerMipmapMode::eLinear)
        .setMinLod(0.0f)
        .setMaxLod(128.0f);

    auto [result, sampler] = device.createSampler(info);
    vk_expect(result, "Failed to create sampler");
    return sampler;
}

void TextureManager::create_samplers(vk::Device device) {
    std::array types = {
        Sampler::NearestRepeat,
        Sampler::NearestMirrored,
        Sampler::NearestClamp,
        Sampler::LinearRepeat,
        Sampler::LinearMirrored,
        Sampler::LinearClamp
    };
    for (auto type: types) {
        this->samplers[get_sampler_index(type)] = create_sampler(device, type);
    }
    this->bind_samplers(this->device, this->desc_set, types);
}

void TextureManager::bind_samplers(
    vk::Device device, 
    vk::DescriptorSet set, 
    std::span<Sampler> types
) {
    std::vector<vk::DescriptorImageInfo> infos;
    infos.reserve(types.size());
    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(types.size());

    for (auto type: types) {
        uint32_t index = get_sampler_index(type);
        vk::Sampler sampler = this->samplers[index];
        if (sampler == vk::Sampler()) {
            continue;
        }

        infos.push_back(vk::DescriptorImageInfo().setSampler(sampler));
        writes.push_back(
            vk::WriteDescriptorSet()
                .setDstSet(set)
                .setDstBinding(1)
                .setDstArrayElement(index)
                .setDescriptorType(vk::DescriptorType::eSampler)
                .setImageInfo(infos.back())
        );
    }
    device.updateDescriptorSets(writes, {});
}

void TextureManager::destroy_samplers(vk::Device device) {
    for (auto sampler: this->samplers) {
        if (sampler != vk::Sampler()) {
            device.destroySampler(sampler);
        }
    }
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
    std::array types = {
        TextureFallback::ColorWhite,
        TextureFallback::ColorError,
        TextureFallback::Normal,
        TextureFallback::AoRoughnessMetallic
    };
    for (auto type: types) {
        uint32_t index = get_fallback_index(type);
        Image image = create_fallback_image(type);
        this->fallbacks[index] = this->create_texture(uploader, image).value();
    }
}