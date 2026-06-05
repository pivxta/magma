#include "materialmanager.h"
#include "texturemanager.h"
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

struct MaterialData {
    TextureIndices base_color;
    TextureIndices normal_map;
    TextureIndices displacement_map;
    TextureIndices aorm_map;

    alignas(16) glm::vec3 base_color_factor;
    float normal_factor;
    float displacement_factor;
    float roughness_factor;
    float metallic_factor;

    float ior;
};

static bool material_has_texture(const Material& material, const TextureId& texture) {
    return material.base_color_texture == texture
        || material.normal_map == texture
        || material.displacement_map == texture
        || material.ao_roughness_metallic_map == texture;
}

static MaterialData get_material_data(const Material& material, const TextureManager& textures) {
    TextureIndices base_color_indices = material.base_color_texture.has_value() ?
        textures.get(*material.base_color_texture, TextureFallback::ColorError) :
        textures.get_fallback(TextureFallback::ColorWhite);

    TextureIndices normal_map_indices = material.normal_map.has_value() ?
        textures.get(*material.normal_map, TextureFallback::Normal) :
        textures.get_fallback(TextureFallback::Normal);

    TextureIndices displacement_map_indices = material.displacement_map.has_value() ?
        textures.get(*material.displacement_map, TextureFallback::Displacement) :
        textures.get_fallback(TextureFallback::Displacement);

    TextureIndices aorm_map_indices = material.ao_roughness_metallic_map.has_value() ?
        textures.get(*material.ao_roughness_metallic_map, TextureFallback::AoRoughnessMetallic) :
        textures.get_fallback(TextureFallback::AoRoughnessMetallic);

    float displacement_factor = 0.0f;
    if (material.displacement_map.has_value()) {
        displacement_factor = material.displacement_factor;
    }

    return {
        .base_color = base_color_indices, 
        .normal_map = normal_map_indices, 
        .displacement_map = displacement_map_indices,
        .aorm_map = aorm_map_indices,
        .base_color_factor = material.base_color_factor,
        .normal_factor = material.normal_factor,
        .displacement_factor = displacement_factor,
        .roughness_factor = material.roughness_factor,
        .metallic_factor = material.metallic_factor,
        .ior = material.ior,
    };
}

/*
static vk::DescriptorPool create_descriptor_pool(vk::Device device, uint32_t frames_in_flight) {
    std::array pool_sizes = {
        vk::DescriptorPoolSize()
            .setType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(frames_in_flight),
    };
    auto [result, descriptor_pool] = device.createDescriptorPool(
        vk::DescriptorPoolCreateInfo()
            .setMaxSets(frames_in_flight)
            .setPoolSizes(pool_sizes)
    );
    vk_expect(result, "Failed to create descriptor pool");
    return descriptor_pool;
}

static vk::DescriptorSetLayout create_set_layout(vk::Device device) {
    std::array bindings = {
        vk::DescriptorSetLayoutBinding()
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment),
    };
    auto [result1, layout] = device.createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo().setBindings(bindings)
    );
    vk_expect(result1, "Failed to create descriptor set layout");
    return layout;
}*/

template<typename T>
static std::vector<vk::DescriptorSet> create_sets(
    vk::Device device,
    vk::DescriptorPool pool,
    vk::DescriptorSetLayout layout,
    const std::vector<Buffer<T>>& buffers
) {
    std::vector<vk::DescriptorSetLayout> layouts;
    layouts.resize(buffers.size(), layout);
    
    auto [result2, sets] = device.allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo()
            .setDescriptorSetCount(static_cast<uint32_t>(buffers.size()))
            .setDescriptorPool(pool)
            .setSetLayouts(layouts)
    );
    vk_expect(result2, "Failed to create descriptor set layout");

    std::vector<vk::WriteDescriptorSet> writes;
    for (uint32_t index = 0; index < sets.size(); index++) {
        vk::DescriptorBufferInfo info = buffers[index].descriptor_info();
        writes.push_back(
            vk::WriteDescriptorSet()
                .setDstSet(sets[index])
                .setDstBinding(0)
                .setDstArrayElement(0)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setBufferInfo(info)
        );
    }
    device.updateDescriptorSets(writes, {});

    return sets;
}

template<typename T>
static std::vector<Buffer<T>> create_buffers(
    vk::Device device,
    vma::Allocator allocator,
    uint32_t frames_in_flight,
    uint32_t max_materials
) {
    std::vector<Buffer<T>> buffers;
    buffers.reserve(frames_in_flight);
    for (uint32_t i = 0; i < frames_in_flight; i++) {
        buffers.push_back(create_mapped_buffer_bda<T>(
            device,
            allocator, 
            vk::BufferUsageFlagBits::eStorageBuffer 
                | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            max_materials
        ));
    }
    return std::move(buffers);
}

MaterialManager::MaterialManager(
    vk::Device device, 
    vma::Allocator allocator, 
    uint32_t frames_in_flight,
    uint32_t max_materials
):
    device(device),
    allocator(allocator),
    dirty(frames_in_flight),
    materials(max_materials)
{
    /*
    this->desc_pool = create_descriptor_pool(device, frames_in_flight);
    this->desc_set_layout = create_set_layout(device);
    */
    this->buffers = create_buffers<MaterialData>(
        device, 
        allocator, 
        frames_in_flight, 
        max_materials
    );
    //this->desc_sets = create_sets(device, this->desc_pool, this->desc_set_layout, this->buffers);
    this->fallback = this->add(Material {});
}

void MaterialManager::destroy() {
    /*
    this->device.destroyDescriptorSetLayout(this->desc_set_layout);
    this->device.destroyDescriptorPool(this->desc_pool);
    */
    for (auto& buffer: this->buffers) {
        buffer.destroy(this->allocator);
    }
}

static SlotKey<Material> get_slot_key(const MaterialId& id) {
    return SlotKey<Material>{
        .index = id.index,
        .generation = id.generation
    };
}

void MaterialManager::flag_dirty_materials(const TextureManager& texture_manager) {
    const auto& updated_textures = texture_manager.get_updated();
    if (updated_textures.empty()) {
        return;
    }
    this->materials.for_each([&](SlotKey<Material> key, const Material& material) {
        bool uses_updated_texture = false;
        for (auto updated_texture: updated_textures) {
            if (material_has_texture(material, updated_texture)) {
                uses_updated_texture = true;
                break;
            }
        }
        if (uses_updated_texture) {
            this->set_dirty(key);
        }
    });
}

void MaterialManager::update_dirty(const TextureManager& texture_manager, uint32_t frame_index) {
    if (this->dirty[frame_index].empty()) {
        return;
    }

    vk::DeviceSize write_start = this->buffers[frame_index].length;
    vk::DeviceSize write_end = 0;
    MaterialData* mapped_data = this->buffers[frame_index].get_mapped();
    for (auto key: this->dirty[frame_index]) {
        write_start = std::min(write_start, static_cast<vk::DeviceSize>(key.index));
        write_end = std::max(write_end, static_cast<vk::DeviceSize>(key.index) + 1);

        if (auto material = this->materials.get(key); material != nullptr) {
            mapped_data[key.index] = get_material_data(*material, texture_manager);
        } else {
            auto fallback_key = get_slot_key(this->fallback);
            if (auto material = this->materials.get(fallback_key); material != nullptr) {
                mapped_data[key.index] = get_material_data(*material, texture_manager);
            }
        }
    }
    this->dirty[frame_index].clear();
    this->buffers[frame_index].flush(this->allocator, write_start, write_end - write_start);
}

MaterialId MaterialManager::add(const Material& material) {
    auto material_key = this->materials.insert(material);
    if (!material_key.has_value()) {
        return this->fallback;
    }
    this->set_dirty(*material_key); 

    MaterialId id = {
        .index = material_key->index,
        .generation = material_key->generation
    };
    
    return id;
}

const Material* MaterialManager::get(MaterialId id) const {
    return this->materials.get(get_slot_key(id));
}

void MaterialManager::set(MaterialId id, const Material& material) {
    SlotKey key = get_slot_key(id);
    if (auto slot = this->materials.get(key); slot != nullptr) {
        *slot = material;
        this->set_dirty(key);
    }
}

void MaterialManager::free(MaterialId id) {
    SlotKey key = get_slot_key(id);
    this->materials.free(key, [&](Material&){
        this->set_dirty(key);
    });
}

void MaterialManager::set_dirty(SlotKey<Material> key) {
    for (auto& dirty: this->dirty) {
        dirty.push_back(key);
    }
}