import surface;
import light;
import brdf;
import cooktorrance;
import parallax;
import textures;
import pbr_material;

struct Vertex {
    float3 position;
    float3 normal;
    float4 tangent;
    float2 tex_coords;
}

struct VertexOutput {
    float4 clip_position: SV_Position;
    float2 tex_coords;
    float3 position;
    float3 normal;
    float3 tangent;
    nointerpolation float handedness;
    nointerpolation uint32_t material_index;
}

struct Instance {
    float3x4 local_to_world;
    uint32_t material_index;
}

struct Draw {
    ImmutablePtr<Vertex> vertices;
    uint instance_offset;
}

struct View {
    float4x4 world_to_clip;
    float3 position;
}

struct Lighting {
    float3 ambient_color;
    float ambient_illuminance;
    uint point_count;
    uint directional_count;
};

struct DrawConstants {
    ImmutablePtr<View> view;
    ImmutablePtr<Lighting> light;
    ImmutablePtr<PointLight> point_lights;
    ImmutablePtr<DirectionalLight> directional_lights;
    ImmutablePtr<Material> materials;
    ImmutablePtr<Instance> instances;
    ImmutablePtr<Draw> draws;
}

[[vk::push_constant]] 
DrawConstants constants;

[shader("vertex")]
VertexOutput vertex_main(
    uint draw_id: SV_DrawIndex,
    uint instance_id: SV_InstanceID,
    uint vertex_id: SV_VertexID
) {
    Instance instance = constants.instances[constants.draws[draw_id].instance_offset + instance_id];
    Vertex vertex = constants.draws[draw_id].vertices[vertex_id];
    var world_position = float3(mul(instance.local_to_world, float4(vertex.position, 1.0)));
    var world_normal = float3(mul(instance.local_to_world, float4(vertex.normal, 0.0)));
    var world_tangent = float3(mul(instance.local_to_world, float4(vertex.tangent.xyz, 0.0)));
    
    VertexOutput output;
    output.clip_position = mul(constants.view.world_to_clip, float4(world_position, 1.0));
    output.position = world_position;
    output.normal = world_normal;
    output.tangent = world_tangent;
    output.handedness = vertex.tangent.w;
    output.tex_coords = vertex.tex_coords;
    output.material_index = instance.material_index;
    return output;
}

typealias FragmentInput = VertexOutput;

[shader("fragment")]
float4 fragment_main(FragmentInput input) : SV_Target {
    let props = constants.materials[input.material_index].sample(
        constants.view.position,
        input.position,
        input.normal,
        input.tangent,
        input.handedness,
        input.tex_coords
    );
    let fragment = props._0;
    let brdf = props._1;

    var color = brdf.calculate_ambient(
        fragment, 
        normalize(constants.view.position - fragment.position),
        constants.light.ambient_color * constants.light.ambient_illuminance
    );

    for (uint i = 0; i < constants.light.directional_count; i++) {
        color += calculate_radiance(
            constants.directional_lights[i], 
            brdf,
            fragment,
            constants.view.position
        );
    }
    
    for (uint i = 0; i < constants.light.point_count; i++) {
        color += calculate_radiance(
            constants.point_lights[i], 
            brdf,
            fragment,
            constants.view.position
        );
    }

    return float4(color, 1.0);
}