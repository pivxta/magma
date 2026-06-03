#include "materialmanager.h"
#include "texturemanager.h"

struct MaterialData {
    TextureIndices base_color;
    TextureIndices normal_map;
    TextureIndices aorm_map;
    alignas(16) glm::vec3 base_color_factor;
    float normal_factor;
    float roughness_factor;
    float metallic_factor;
    float ior;
};

static std::vector<const TextureId*> get_material_textures(const Material& material) {
    std::vector<const TextureId*> textures;
    if (material.base_color_texture.has_value()) {
        textures.push_back(&*material.base_color_texture);
    }
    if (material.normal_map.has_value()) {
        textures.push_back(&*material.normal_map);
    }
    if (material.ao_roughness_metallic_map.has_value()) {
        textures.push_back(&*material.ao_roughness_metallic_map);
    }
    return textures;
}

static MaterialData get_material_data(const Material& material, const TextureManager& textures) {
    TextureIndices base_color_indices = material.base_color_texture.has_value() ?
        textures.get(*material.base_color_texture, TextureFallback::ColorError) :
        textures.get_fallback(TextureFallback::ColorWhite);

    TextureIndices normal_map_indices = material.normal_map.has_value() ?
        textures.get(*material.normal_map, TextureFallback::Normal) :
        textures.get_fallback(TextureFallback::Normal);

    TextureIndices aorm_map_indices = material.ao_roughness_metallic_map.has_value() ?
        textures.get(*material.normal_map, TextureFallback::AoRoughnessMetallic) :
        textures.get_fallback(TextureFallback::AoRoughnessMetallic);

    return {
        .base_color = base_color_indices, 
        .normal_map = normal_map_indices, 
        .aorm_map = aorm_map_indices,
        .base_color_factor = material.base_color_factor,
        .normal_factor = material.normal_factor,
        .roughness_factor = material.roughness_factor,
        .metallic_factor = material.metallic_factor,
        .ior = material.ior,
    };
}

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
}

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
    vma::Allocator allocator,
    uint32_t frames_in_flight,
    uint32_t max_materials
) {
    std::vector<Buffer<T>> buffers;
    for (uint32_t i = 0; i < frames_in_flight; i++) {
        buffers.push_back(create_mapped_buffer<T>(
            allocator, 
            vk::BufferUsageFlagBits::eStorageBuffer,
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
    this->desc_pool = create_descriptor_pool(device, frames_in_flight);
    this->desc_set_layout = create_set_layout(device);
    this->buffers = create_buffers<MaterialData>(allocator, frames_in_flight, max_materials);
    this->desc_sets = create_sets(device, this->desc_pool, this->desc_set_layout, this->buffers);
    this->fallback = this->add(Material {});
}

void MaterialManager::destroy() {
    this->device.destroyDescriptorSetLayout(this->desc_set_layout);
    this->device.destroyDescriptorPool(this->desc_pool);
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
    for (const auto& texture: texture_manager.get_updated()) {
        if (!this->dependency_map.contains(texture)) {
            continue;
        }
        for (const auto& [material, count]: this->dependency_map[texture]) {
            SlotKey key = get_slot_key(material);
            if (auto slot = this->materials.get(key); slot != nullptr) {
                this->set_dirty(key);
            }
        }
    }
}

void MaterialManager::update_dirty(const TextureManager& texture_manager, uint32_t frame_index) {
    if (this->dirty[frame_index].empty()) {
        return;
    }

    uint32_t write_start = this->buffers[frame_index].length;
    uint32_t write_end = 0;
    MaterialData* mapped_data = this->buffers[frame_index].get_mapped();
    for (auto key: this->dirty[frame_index]) {
        write_start = std::min(write_start, key.index);
        write_end = std::max(write_end, key.index + 1);

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
    for (auto texture: get_material_textures(material)) {
        this->add_dependency(id, *texture);
    }

    return id;
}

const Material* MaterialManager::get(MaterialId id) const {
    return this->materials.get(get_slot_key(id));
}

void MaterialManager::set(MaterialId id, const Material& material) {
    SlotKey key = get_slot_key(id);
    if (auto slot = this->materials.get(key); slot != nullptr) {
        for (auto texture: get_material_textures(*slot)) {
            this->remove_dependency(id, *texture);
        }
        for (auto texture: get_material_textures(material)) {
            this->add_dependency(id, *texture);
        }
        *slot = material;
    }
    this->set_dirty(key);
}

void MaterialManager::free(MaterialId id) {
    SlotKey key = get_slot_key(id);
    if (auto slot = this->materials.get(key); slot != nullptr) {
        for (auto texture: get_material_textures(*slot)) {
            this->remove_dependency(id, *texture);
        }
    }
    this->materials.free(get_slot_key(id), [](Material&){});
    this->set_dirty(key);
}

void MaterialManager::set_dirty(SlotKey<Material> key) {
    for (auto& dirty: this->dirty) {
        dirty.push_back(key);
    }
}

void MaterialManager::add_dependency(MaterialId material, TextureId texture) {
    if (!this->dependency_map.contains(texture)) {
        std::unordered_map<MaterialId, uint32_t> count;
        count[material] = 1;
        this->dependency_map[texture] = std::move(count);
    } else {
        auto& count = this->dependency_map[texture];
        if (count.contains(material)) {
            count[material] += 1;
        } else {
            count[material] = 1;
        }
    }
}

void MaterialManager::remove_dependency(MaterialId material, TextureId texture) {
    if (!this->dependency_map.contains(texture)) {
        return;
    }

    auto& count = this->dependency_map[texture];
    if (!count.contains(material)) {
        return;
    } 

    auto& count_value = count[material];
    if (count_value <= 1) {
        count.erase(material);
    } else {
        count_value -= 1;
    }
}