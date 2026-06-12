#include "mesh_manager.h"

static constexpr vk::DeviceSize VERTEX_ALIGNMENT = 16;
static constexpr vk::DeviceSize INDEX_ALIGNMENT = sizeof(uint32_t);

MeshManager::MeshManager(
    const DeviceHandle& device,
    uint32_t frames_in_flight,
    vk::DeviceSize vertex_heap_capacity,
    vk::DeviceSize index_heap_capacity
) {
    this->vertex_heap = HeapBuffer(
        device,
        frames_in_flight,
        VERTEX_ALIGNMENT,
        vk::BufferCreateInfo()
            .setSize(vertex_heap_capacity)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setUsage(
                vk::BufferUsageFlagBits::eStorageBuffer
                    | vk::BufferUsageFlagBits::eShaderDeviceAddress
                    | vk::BufferUsageFlagBits::eTransferDst
            ),
        vma::AllocationCreateInfo()
            .setUsage(vma::MemoryUsage::eGpuOnly)
    );
    this->index_heap = HeapBuffer(
        device,
        frames_in_flight,
        INDEX_ALIGNMENT,
        vk::BufferCreateInfo()
            .setSize(index_heap_capacity)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setUsage(
                vk::BufferUsageFlagBits::eIndexBuffer 
                    | vk::BufferUsageFlagBits::eTransferDst
            ),
        vma::AllocationCreateInfo()
            .setUsage(vma::MemoryUsage::eGpuOnly)
    );
}

SlotKey<MeshManager::MeshSubBuffers> MeshManager::get_slot_key(MeshId id) {
    return {
        .index = id.index,
        .generation = id.generation
    };
}

MeshId MeshManager::reserve() {
    SlotKey<MeshSubBuffers> key = this->meshes.insert({}).value();
    return MeshId{
        .index = key.index,
        .generation = key.generation
    };
}

MeshId MeshManager::add(Uploader& uploader, const Mesh& mesh) {
    MeshId id = this->reserve();
    if (auto sub_buffers = this->create_sub_buffers(uploader, mesh)) {
        *this->meshes.get(get_slot_key(id)) = sub_buffers.value();
    }
    return id;
}

MeshData MeshManager::get(MeshId id) const {
    if (auto sub_buffers = this->meshes.get(get_slot_key(id)); sub_buffers != nullptr) {
        assert(sub_buffers->indices.buffer_offset() % sizeof(uint32_t) == 0);
        auto index_offset = sub_buffers->indices.buffer_offset() / sizeof(uint32_t);
        return {
            .vertices_address = sub_buffers->vertices.address(), 
            .index_offset = static_cast<uint32_t>(index_offset),
            .index_count = static_cast<uint32_t>(sub_buffers->indices.length()),
        };
    }
    return {
        .vertices_address = 0,
        .index_offset = 0,
        .index_count = 0
    };
}

bool MeshManager::set(MeshId id, Uploader& uploader, const Mesh& mesh) {
    if (auto sub_buffers = this->meshes.get(get_slot_key(id)); sub_buffers != nullptr) {
        if (auto new_sub_buffers = this->create_sub_buffers(uploader, mesh); 
            new_sub_buffers.has_value()) 
        {
            this->vertex_heap.deferred_free(sub_buffers->vertices);
            this->index_heap.deferred_free(sub_buffers->indices);
            *sub_buffers = new_sub_buffers.value();
            return true;
        }
        return false;
    }
    return false;
}

bool MeshManager::free(MeshId id) {
    SlotKey key = get_slot_key(id);
    return this->meshes.free(key, [&](MeshSubBuffers& sub_buffers) {
        this->vertex_heap.deferred_free(sub_buffers.vertices);
        this->index_heap.deferred_free(sub_buffers.indices);
    });
}

bool MeshManager::is_valid(MeshId id) const {
    return this->meshes.is_valid(get_slot_key(id));
}

void MeshManager::free_pending() {
    this->vertex_heap.free_pending();
    this->index_heap.free_pending();
}

void MeshManager::begin_frame(uint64_t frame_counter) {
    this->vertex_heap.begin_frame(frame_counter);
    this->index_heap.begin_frame(frame_counter);
}

static std::optional<std::vector<VertexData>> interlace_mesh_attributes(const Mesh& mesh) {
    if (mesh.positions.size() != mesh.normals.size()
        || mesh.positions.size() != mesh.tex_coords.size()
        || mesh.positions.size() != mesh.tangents.size())
    {
        return std::nullopt;
    }

    std::vector<VertexData> vertices;
    vertices.reserve(mesh.positions.size());
    for (size_t i = 0; i < mesh.positions.size(); i++) {
        vertices.push_back({
            .position = mesh.positions[i],
            .normal = mesh.normals[i],
            .tangent = mesh.tangents[i],
            .tex_coords = mesh.tex_coords[i]
        });
    }

    return vertices;
}

std::optional<MeshManager::MeshSubBuffers> MeshManager::create_sub_buffers(
    Uploader& uploader, 
    const Mesh& mesh
) {
    auto vertices = interlace_mesh_attributes(mesh);
    const auto& indices = mesh.indices;

    if (!vertices.has_value()) {
        return std::nullopt;
    }

    auto vertices_buffer = this->vertex_heap.allocate<VertexData>(vertices.value().size());
    if (!vertices_buffer.has_value()) {
        return std::nullopt;
    }

    auto indices_buffer = this->index_heap.allocate<uint32_t>(indices.size());
    if (!indices_buffer.has_value()) {
        return std::nullopt;
    }

    uploader.upload_buffer(
        BufferUpload()
            .set_buffer(vertices_buffer->buffer())
            .set_offset(vertices_buffer->buffer_offset())
            .set_memory(vertices.value())
            .set_usage_access(vk::AccessFlagBits2::eShaderRead)
            .set_usage_stage(vk::PipelineStageFlagBits2::eVertexShader)
    );
    uploader.upload_buffer(
        BufferUpload()
            .set_buffer(indices_buffer->buffer())
            .set_offset(indices_buffer->buffer_offset())
            .set_memory(indices)
            .set_usage_access(vk::AccessFlagBits2::eIndexRead)
            .set_usage_stage(vk::PipelineStageFlagBits2::eIndexInput)
    );

    return MeshSubBuffers{
        .vertices = vertices_buffer.value(),
        .indices = indices_buffer.value()
    };
}