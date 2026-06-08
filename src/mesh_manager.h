#pragma once
#include "heap_buffer.h"
#include "slot_map.h"
#include "resource.h"
#include "uploader.h"
#include "mesh.h"
#include <glm/glm.hpp>

struct MeshData {
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
    MeshManager(
        vk::Device device, 
        vma::Allocator allocator,
        uint32_t frames_in_flight,
        vk::DeviceSize vertex_heap_capacity,
        vk::DeviceSize index_heap_capacity
    );
    void destroy();

    MeshData get(MeshId id) const;
    MeshId reserve();
    MeshId add(Uploader& uploader, const Mesh& mesh);
    bool set(MeshId id, Uploader& uploader, const Mesh& mesh);
    bool free(MeshId id);

    void free_pending();
    void begin_frame(uint64_t frame_counter);

    const Buffer& index_buffer() const {
        return this->index_heap.buffer();
    }

private:
    struct MeshSubBuffers {
        HeapSubBuffer<VertexData> vertices;
        HeapSubBuffer<uint32_t> indices;
    };

    static SlotKey<MeshSubBuffers> get_slot_key(MeshId id);
    std::optional<MeshSubBuffers> create_sub_buffers(Uploader& uploader, const Mesh& mesh);

    HeapBuffer vertex_heap;
    HeapBuffer index_heap;

    SlotMap<MeshSubBuffers> meshes;
};