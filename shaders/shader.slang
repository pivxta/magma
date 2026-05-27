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
}

struct VertexOutput {
    float4 sv_position : SV_Position;
    float3 position;
    float3 normal;
    float2 tex_coords;
};

struct Uniforms {
    float4x4 world_to_clip;
    float3 view_position;
    float3 ambient_light_color;
    uint point_lights;
    uint directional_lights;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniforms> uniforms;

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
    output.sv_position = mul(uniforms.world_to_clip, float4(world_position, 1.0));
    output.position = world_position;
    output.normal = world_normal;
    output.tex_coords = vertex.tex_coords;
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

[[vk::binding(1, 0)]]
StructuredBuffer<DirectionalLight> directional_lights;
[[vk::binding(2, 0)]]
StructuredBuffer<PointLight> point_lights;
[[vk::binding(3, 0)]]
Sampler2D sampler;

static const float PI = 3.14159265358979323846264338;

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target {
    var albedo = sampler.Sample(input.tex_coords).rgb;
    var color = albedo * uniforms.ambient_light_color;
    for (uint i = 0; i < uniforms.directional_lights; i++) {
        color += directional_lights[i].illuminate(
            uniforms.view_position,
            input.position,
            input.normal,
            albedo,
        );
    }
    for (uint i = 0; i < uniforms.point_lights; i++) {
        color += point_lights[i].illuminate(
            uniforms.view_position,
            input.position,
            input.normal,
            albedo,
        );
    }
    return float4(color, 1.0);
}