#pragma once
#include "heap_buffer.h"
#include "slot_map.h"
#include "resource.h"
#include "uploader.h"
#include "mesh.h"
#include <glm/glm.hpp>

struct MeshDrawData {
    vk::DeviceAddress vertices_address;
    uint32_t index_offset;
    uint32_t index_count;
};

struct VertexData {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 tangent;
    glm::vec2 tex_coords;
};

class MeshManager {
public:
    MeshManager() = default;

    explicit MeshManager(
        const DeviceHandle& device,
        vk::DeviceSize vertex_heap_capacity,
        vk::DeviceSize index_heap_capacity
    );

    MeshManager(const MeshManager&) = delete;
    MeshManager& operator=(const MeshManager&) = delete;
    MeshManager(MeshManager&&) noexcept = default;
    MeshManager& operator=(MeshManager&&) noexcept = default;

    MeshDrawData get_draw_data(MeshId id) const;
    Aabb get_bounds(MeshId id) const;
    MeshId reserve();
    MeshId add(Uploader& uploader, const Mesh& mesh);
    bool set(MeshId id, Uploader& uploader, const Mesh& mesh);
    bool free(MeshId id);

    bool is_valid(MeshId id) const;

    const Buffer& index_buffer() const {
        return this->index_heap.buffer();
    }

private:
    struct MeshData {
        HeapSubBuffer<VertexData> vertices;
        HeapSubBuffer<uint32_t> indices;
        Aabb bounds;
    };

    static SlotKey<MeshData> get_slot_key(MeshId id);
    std::optional<MeshData> create_data(Uploader& uploader, const Mesh& mesh);

    HeapBuffer vertex_heap;
    HeapBuffer index_heap;

    SlotMap<MeshData> meshes;
};