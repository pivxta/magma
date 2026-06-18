#pragma once
#include "resource.h"
#include "slot_map.h"
#include "material.h"
#include "device.h"
#include "buffer.h"
#include <vector>
#include <vulkan/vulkan.hpp>

struct MaterialData;
class TextureManager;

class MaterialManager {
public:
    MaterialManager() = default;

    explicit MaterialManager(DeviceHandle device, uint32_t max_materials);

    MaterialManager(const MaterialManager&) = delete;
    MaterialManager& operator=(const MaterialManager&) = delete;
    MaterialManager(MaterialManager&&) noexcept = default;
    MaterialManager& operator=(MaterialManager&&) noexcept = default;
    ~MaterialManager();

    void update_dirty(const TextureManager& texture_manager);

    const Material* get(MaterialId id) const;
    MaterialId add(const Material& material);
    void set(MaterialId id, const Material& material);
    void free(MaterialId id);

    uint32_t get_index(MaterialId id) const;

    vk::DeviceAddress buffer_address() const {
        return this->buffers[this->device->frame_index()].address();
    }

private:
    void flag_dirty_materials(const TextureManager& texture_manager);
    void set_dirty(SlotKey<Material> key);

    DeviceHandle device;
    std::vector<Buffer> buffers;
    std::vector<std::vector<SlotKey<Material>>> dirty;
    SlotMap<Material> materials;
    MaterialId fallback;
};