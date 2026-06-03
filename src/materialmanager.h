#pragma once
#include "resource.h"
#include "slotmap.h"
#include "material.h"
#include "buffer.h"
#include <vector>
#include <vulkan/vulkan.hpp>

struct MaterialData;
class TextureManager;

class MaterialManager {
public:
    MaterialManager() {}
    MaterialManager(
        vk::Device device, 
        vma::Allocator allocator, 
        uint32_t frames_in_flight,
        uint32_t max_materials
    );
    void destroy();

    void flag_dirty_materials(const TextureManager& texture_manager);
    void update_dirty(const TextureManager& texture_manager, uint32_t frame_index);

    MaterialId add(const Material& material);
    const Material* get(MaterialId id) const;
    void set(MaterialId id, const Material& material);
    void free(MaterialId id);

    vk::DescriptorSetLayout descriptor_set_layout() const {
        return this->desc_set_layout;
    }

    vk::DescriptorSet descriptor_set(uint32_t frame_index) const {
        return this->desc_sets[frame_index];
    }

private:
    void set_dirty(SlotKey<Material> key);
    void add_dependency(MaterialId material, TextureId texture);
    void remove_dependency(MaterialId material, TextureId texture);

    vk::Device device;
    vma::Allocator allocator;

    vk::DescriptorPool desc_pool;
    vk::DescriptorSetLayout desc_set_layout;
    std::vector<vk::DescriptorSet> desc_sets;

    std::unordered_map<TextureId, std::unordered_map<MaterialId, uint32_t>> dependency_map;
    std::vector<Buffer<MaterialData>> buffers;
    std::vector<std::vector<SlotKey<Material>>> dirty;
    SlotMap<Material> materials;
    MaterialId fallback;
};