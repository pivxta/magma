struct Vertex {
    float3 position;
    float3 normal;
    float4 tangent;
    float2 tex_coords;
}

struct Instance {
    float3x4 local_to_world;
    uint32_t material_index;
}

struct Draw {
    ImmutablePtr<Vertex> vertices;
    uint instance_offset;
}

struct ShadowConstants {
    uint view_count;
    ImmutablePtr<float4x4> views;
    ImmutablePtr<Instance> instances;
    ImmutablePtr<Draw> draws;
}

[[vk::push_constant]]
ShadowConstants constants;

[shader("vertex")]
float4 vertex_main(
    uint draw_id: SV_DrawIndex,
    uint view_id: SV_ViewID,
    uint instance_id: SV_InstanceID,
    uint vertex_id: SV_VertexID
) : SV_Position {
    Instance instance = constants.instances[constants.draws[draw_id].instance_offset + instance_id];
    Vertex vertex = constants.draws[draw_id].vertices[vertex_id];
    float3 world_position = float3(mul(instance.local_to_world, float4(vertex.position, 1.0)));
    return mul(constants.views[view_id], float4(world_position, 1.0));
}
