struct VertexInput {
    float3 position;
    float3 normal;
    float2 tex_coords;
}

struct InstanceInput {
    float3 local_to_world_x;
    float3 local_to_world_y;
    float3 local_to_world_z;
    float3 translation;
    uint32_t material_index;
}

struct VertexOutput {
    float4 sv_position : SV_Position;
    float3 position;
    float3 normal;
    float2 tex_coords;
    nointerpolation uint32_t material_index;
};

struct ViewUniforms {
    float4x4 world_to_clip;
};

[[vk::binding(0, 0)]]
ConstantBuffer<ViewUniforms> view;

[shader("vertex")]
VertexOutput vertexMain(VertexInput vertex, InstanceInput instance) {
    float3 world_position =
        instance.local_to_world_x * vertex.position.x
            + instance.local_to_world_y * vertex.position.y
            + instance.local_to_world_z * vertex.position.z
            + instance.translation;
    // The scaling is uniform so this is fine
    float3 world_normal =
        instance.local_to_world_x * vertex.normal.x
            + instance.local_to_world_y * vertex.normal.y
            + instance.local_to_world_z * vertex.normal.z;
    
    VertexOutput output;
    output.sv_position = mul(view.world_to_clip, float4(world_position, 1.0));
    output.position = world_position;
    output.normal = world_normal;
    output.tex_coords = vertex.tex_coords;
    output.material_index = instance.material_index;
    return output;
}

typealias FragmentInput = VertexOutput;

interface Light {
    float3 illuminate(
        float3 view,
        float3 position,
        float3 normal,
        float3 albedo,
    );
};

struct DirectionalLight: Light {
    float3 direction;
    float padding;
    float3 color;
    float illuminance;

    float3 illuminate(
        float3 view,
        float3 position,
        float3 normal,
        float3 albedo,
    ) {
        return this.illuminance 
            * this.color 
            * albedo
            * max(dot(normalize(-this.direction), normal), 0.0);
    }
};

struct PointLight: Light {
    float3 position;
    float radius;
    float3 color;
    float intensity;
    
    float3 illuminate(
        float3 view,
        float3 position,
        float3 normal,
        float3 albedo,
    ) {
        var to_light = this.position - position;
        var lambert = max(dot(normalize(to_light), normal), 0.0);
        var attenuation = 1.0 / (4.0 * PI * dot(to_light, to_light));
        return this.intensity * this.color * albedo * lambert * attenuation;
    }
};

struct Material {
    uint32_t albedo_index;
    uint32_t albedo_sampler_index;
};

struct LightUniforms {
    float3 view_position;
    float3 ambient_color;
    float ambient_illuminance;
    uint point_count;
    uint directional_count;
};

[[vk::binding(1, 0)]]
ConstantBuffer<LightUniforms> light;
[[vk::binding(2, 0)]]
StructuredBuffer<DirectionalLight> directional_lights;
[[vk::binding(3, 0)]]
StructuredBuffer<PointLight> point_lights;

[[vk::binding(0, 1)]]
Texture2D[] textures;
[[vk::binding(1, 1)]]
SamplerState[] samplers;
[[vk::binding(2, 1)]]
StructuredBuffer<Material> materials;

static const float PI = 3.14159265358979323846264338;

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target {
    var material = materials[input.material_index];
    var albedo_texture = textures[material.albedo_index];
    var albedo_sampler = samplers[material.albedo_sampler_index];

    var albedo = albedo_texture.Sample(albedo_sampler, input.tex_coords).rgb;
    var color = albedo * light.ambient_color * light.ambient_illuminance;
    for (uint i = 0; i < light.directional_count; i++) {
        color += directional_lights[i].illuminate(
            light.view_position,
            input.position,
            input.normal,
            albedo,
        );
    }
    for (uint i = 0; i < light.point_count; i++) {
        color += point_lights[i].illuminate(
            light.view_position,
            input.position,
            input.normal,
            albedo,
        );
    }
    return float4(color, 1.0);
}