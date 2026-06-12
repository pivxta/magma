#include "material_manager.h"
#include "texture_manager.h"
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

static std::vector<Buffer> create_buffers(
    const DeviceHandle& device,
    uint32_t frames_in_flight,
    uint32_t max_materials
) {
    std::vector<Buffer> buffers;
    buffers.reserve(frames_in_flight);
    for (uint32_t i = 0; i < frames_in_flight; i++) {
        buffers.emplace_back(
            device,
            vk::BufferCreateInfo()
                .setSharingMode(vk::SharingMode::eExclusive)
                .setSize(max_materials * sizeof(MaterialData))
                .setUsage(
                    vk::BufferUsageFlagBits::eStorageBuffer
                    | vk::BufferUsageFlagBits::eShaderDeviceAddress
                ),
            vma::AllocationCreateInfo()
                .setUsage(vma::MemoryUsage::eAuto)
                .setFlags(
                    vma::AllocationCreateFlagBits::eMapped
                    | vma::AllocationCreateFlagBits::eHostAccessRandom
                )
        );
    }
    return buffers;
}

MaterialManager::MaterialManager(
    DeviceHandle device,
    uint32_t frames_in_flight,
    uint32_t max_materials
):
    device(std::move(device)),
    dirty(frames_in_flight),
    materials(max_materials)
{
    this->buffers = create_buffers(
        this->device,
        frames_in_flight,
        max_materials
    );
    this->fallback = this->add(Material {});
}

MaterialManager::~MaterialManager() {
    if (!this->device) {
        return;
    }
    for (auto& buffer: this->buffers) {
        buffer.destroy(this->device);
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

    vk::DeviceSize write_start = this->buffers[frame_index].size();
    vk::DeviceSize write_end = 0;
    auto mapped_data = reinterpret_cast<MaterialData*>(this->buffers[frame_index].mapped());
    for (auto key: this->dirty[frame_index]) {
        auto entry_offset = static_cast<vk::DeviceSize>(key.index) * sizeof(MaterialData);
        auto entry_size = static_cast<vk::DeviceSize>(sizeof(MaterialData));
        write_start = std::min(write_start, entry_offset);
        write_end = std::max(write_end, entry_offset + entry_size);

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
    this->buffers[frame_index].flush(this->device, write_start, write_end - write_start);
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

uint32_t MaterialManager::get_index(MaterialId id) const {
    if (this->materials.is_valid(get_slot_key(id))) {
        return id.index;
    } else {
        return this->fallback.index;
    }
}

const Material* MaterialManager::get(MaterialId id) const {
    if (auto material = this->materials.get(get_slot_key(id)); material != nullptr) {
        return material;
    } else {
        return this->materials.get(get_slot_key(this->fallback));
    }
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