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
    MaterialManager() = default;
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

    vk::DeviceAddress buffer_device_address(uint32_t frame_index) const {
        return this->buffers[frame_index].get_device_address();
    }
    /*
    vk::DescriptorSetLayout descriptor_set_layout() const {
        return this->desc_set_layout;
    }

    vk::DescriptorSet descriptor_set(uint32_t frame_index) const {
        return this->desc_sets[frame_index];
    }
        */

private:
    void set_dirty(SlotKey<Material> key);

    vk::Device device;
    vma::Allocator allocator;

    /*
    vk::DescriptorPool desc_pool;
    vk::DescriptorSetLayout desc_set_layout;
    std::vector<vk::DescriptorSet> desc_sets;
    */

    std::vector<Buffer<MaterialData>> buffers;
    std::vector<std::vector<SlotKey<Material>>> dirty;
    SlotMap<Material> materials;
    MaterialId fallback;
};