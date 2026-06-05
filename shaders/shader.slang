struct Vertex {
    float3 position;
    float3 normal;
    float4 tangent;
    float2 tex_coords;
}

struct Instance {
    float3 local_to_world_x;
    float3 local_to_world_y;
    float3 local_to_world_z;
    float3 translation;
    uint32_t material_index;

    float3 transform_point(float3 point) {
        return this.local_to_world_x * point.x
            + this.local_to_world_y * point.y
            + this.local_to_world_z * point.z
            + this.translation;
    }

    float3 transform_vector(float3 vector) {
        return this.local_to_world_x * vector.x
            + this.local_to_world_y * vector.y
            + this.local_to_world_z * vector.z;
    }

    float3 transform_normal(float3 normal) {
        // The scaling is uniform so this is fine
        return this.local_to_world_x * normal.x
            + this.local_to_world_y * normal.y
            + this.local_to_world_z * normal.z;
    }
}

struct DrawConstants {
    ImmutablePtr<ViewUniforms> view;
    ImmutablePtr<LightUniforms> light;
    ImmutablePtr<PointLight> point_lights;
    ImmutablePtr<DirectionalLight> directional_lights;
    ImmutablePtr<Instance> instances;
    ImmutablePtr<Vertex> vertices;
    ImmutablePtr<Material> materials;
}

[[vk::push_constant]] DrawConstants constants;

struct VertexOutput {
    float4 sv_position : SV_Position;
    float2 tex_coords;
    float3 position;
    float3 normal;
    float3 tangent;
    nointerpolation float handedness;
    nointerpolation uint32_t material_index;
}

struct ViewUniforms {
    float4x4 world_to_clip;
    float3 position;
}

[shader("vertex")]
VertexOutput vertexMain(uint vertex_id: SV_VertexID, uint instance_id: SV_InstanceID) {
    Instance instance = constants.instances[instance_id];
    Vertex vertex = constants.vertices[vertex_id];
    float3 world_position = instance.transform_point(vertex.position);
    float3 world_normal = instance.transform_normal(vertex.normal);
    float3 world_tangent = instance.transform_vector(vertex.tangent.xyz);
    
    VertexOutput output;
    output.sv_position = mul(constants.view.world_to_clip, float4(world_position, 1.0));
    output.position = world_position;
    output.normal = world_normal;
    output.tangent = world_tangent;
    output.handedness = vertex.tangent.w;
    output.tex_coords = vertex.tex_coords;
    output.material_index = instance.material_index;
    return output;
}

static const float PI = 3.14159265358979323846264338;

typealias FragmentInput = VertexOutput;

struct FragmentProperties {
    float3 position;
    float3 geometry_normal;
    float3 normal;
    float3 base_color;
    float ambient_occlusion;
};

interface Brdf {
    float3 calculate(
        FragmentProperties surface_props,
        float3 omega_i,
        float3 omega_o
    );
};

interface Light {
    float3 get_incident_radiance(float3 position);
    float3 get_omega_i(float3 position);
};

struct CookTorranceBrdf : Brdf {
    float alpha;
    float metallic;
    float ior;

    float microfacet_ggx(float wh_dot_n) {
        float alpha2 = this.alpha * this.alpha;
        float k = wh_dot_n * wh_dot_n * (alpha2 - 1.0) + 1.0;
        return alpha2 / (PI * k * k);
    }

    float visibility_smith(float wi_dot_n, float wo_dot_n) {
        let alpha2 = this.alpha * this.alpha;
        let ggx_o = wi_dot_n * sqrt(wo_dot_n * wo_dot_n * (1.0 - alpha2) + alpha2);
        let ggx_i = wo_dot_n * sqrt(wi_dot_n * wi_dot_n * (1.0 - alpha2) + alpha2);
        return 0.5 / (ggx_o + ggx_i);
    }

    float3 fresnel_schlick(float3 f0, float wo_dot_wh) {
        return f0 + (1.0 - f0) * pow(1.0 - wo_dot_wh, 5.0);
    }

    float3 calculate(
        FragmentProperties fragment,
        float3 omega_i,
        float3 omega_o
    ) {
        float3 omega_h = normalize(omega_i + omega_o);
        float wi_dot_n = max(dot(omega_i, fragment.normal), 0.000001);
        float wo_dot_n = max(dot(omega_o, fragment.normal), 0.000001);
        float wh_dot_n = max(dot(omega_h, fragment.normal), 0.0);
        float wo_dot_wh = max(dot(omega_o, omega_h), 0.0);

        float df0_sqrt = (this.ior - 1.0) / (this.ior + 1.0);
        float df0 = df0_sqrt * df0_sqrt;
        float3 f0 = lerp(float3(df0), fragment.base_color, this.metallic);
        float3 f = this.fresnel_schlick(f0, wo_dot_wh);
        float3 v = this.visibility_smith(wi_dot_n, wo_dot_n);
        float d = this.microfacet_ggx(wh_dot_n);
        float3 specular = f * d * v;
        float3 diffuse = (1.0 - this.metallic) * (1.0 - f) * fragment.base_color / PI;
        return specular + diffuse;
    }
}

struct DirectionalLight: Light {
    float3 direction;
    float padding;
    float3 color;
    float illuminance;

    float3 get_incident_radiance(float3 position) {
        return this.illuminance * this.color;
    }

    float3 get_omega_i(float3 position) {
        return normalize(-this.direction);
    }
};

struct PointLight: Light {
    float3 position;
    float radius;
    float3 color;
    float intensity;

    float3 get_incident_radiance(float3 position) {
        let fragment_to_light = this.position - position;
        let distance_sq = dot(fragment_to_light, fragment_to_light);
        return (this.color * this.intensity) / max(4.0 * PI * distance_sq, 0.0001);
    }

    float3 get_omega_i(float3 position) {
        return normalize(this.position - position);
    }
};

float shadow_terminator_mask(float3 omega_i, float3 normal, float3 geometry_normal) {
    float wi_dot_n = max(dot(normal, omega_i), 0.0);
    float wi_dot_g = max(dot(geometry_normal, omega_i), 0.0);
    float g_dot_n = max(dot(geometry_normal, normal), 0.0);

    if (wi_dot_n <= 0.0 || wi_dot_g <= 0.0 || g_dot_n <= 0.0)
        return 0.0;

    float g = min(wi_dot_g / (wi_dot_n * g_dot_n), 1.0);
    return g * (g * (1.0 - g) + 1.0);
}

float3 calculate_radiance<B: Brdf, L: Light>(
    L light,
    B brdf,
    FragmentProperties fragment,
    float3 view_position,
) {
    let omega_i = light.get_omega_i(fragment.position);
    let omega_o = normalize(view_position - fragment.position);
    let shadow_mask = shadow_terminator_mask(omega_i, fragment.normal, fragment.geometry_normal);
    return brdf.calculate(fragment, omega_i, omega_o)
            * light.get_incident_radiance(fragment.position)
            * max(dot(omega_i, fragment.normal), 0.0)
            * shadow_mask;
}

struct TextureIndices {
    uint texture;
    uint sampler;
}

struct MaterialFragmentProperties<B: Brdf> {
    FragmentProperties fragment;
    B brdf;
}

struct Material {
    TextureIndices base_color;
    TextureIndices normal_map;
    TextureIndices displacement_map;
    TextureIndices aorm_map;

    float3 base_color_factor;
    float normal_factor;
    float displacement_factor;
    float roughness_factor;
    float metallic_factor;

    float ior;

    float2 parallax_displace(
        float3 local_view_dir,
        float2 tex_coords,
        float displacement_factor,
        Texture2D displacement_texture,
        SamplerState displacement_sampler
    ) {
        const float min_layers = 4.0;
        const float max_layers = 12.0;

        let num_layers = lerp(max_layers, min_layers, abs(local_view_dir.z));
        let layer_depth = 1.0 / num_layers;
        var current_layer_depth = 0.0;

        let dx = ddx(tex_coords);
        let dy = ddy(tex_coords);

        let delta_tex_coords = (local_view_dir.xy * displacement_factor / local_view_dir.z) / num_layers;
        var current_tex_coords = tex_coords;
        var current_depth = 1.0 - displacement_texture.SampleGrad(
            displacement_sampler, 
            current_tex_coords, 
            dx, dy
        ).r;

        [loop]
        while (current_layer_depth < current_depth) {
            current_tex_coords -= delta_tex_coords;
            current_layer_depth += layer_depth;
            current_depth = 1.0 - displacement_texture.SampleGrad(
                displacement_sampler, 
                current_tex_coords, 
                dx, dy
            ).r;
        }

        let prev_tex_coords = current_tex_coords + delta_tex_coords;
        let after_depth  = current_depth - current_layer_depth;
        let before_depth = 
            (1.0 - displacement_texture.SampleGrad(displacement_sampler, prev_tex_coords, dx, dy).r)
                - current_layer_depth 
                + layer_depth;

        let weight = after_depth / (after_depth - before_depth);
        return lerp(current_tex_coords, prev_tex_coords, weight);
    }

    MaterialFragmentProperties<CookTorranceBrdf> get_fragment_properties(
        float3 view_position,
        float3 position,
        float3 normal,
        float3 tangent,
        float tangent_handedness,
        float2 tex_coords
    ) {
        var geometry_normal = normalize(normal);
        var tangent = normalize(tangent);
        var bitangent = tangent_handedness * cross(geometry_normal, tangent);

        var disp_tex_coords = tex_coords;
        if (this.displacement_factor >= 0.0001) {
            let view_dir = normalize(view_position - position);
            let local_view_dir = float3(
                dot(view_dir, tangent),
                dot(view_dir, bitangent),
                dot(view_dir, geometry_normal)
            );

            let displacement_map_texture = textures[NonUniformResourceIndex(this.displacement_map.texture)];
            let displacement_map_sampler = samplers[NonUniformResourceIndex(this.displacement_map.sampler)];
            disp_tex_coords = parallax_displace(
                local_view_dir,
                tex_coords,
                displacement_factor,
                displacement_map_texture,
                displacement_map_sampler
            );
        }

        var base_color_texture = textures[NonUniformResourceIndex(this.base_color.texture)];
        var base_color_sampler = samplers[NonUniformResourceIndex(this.base_color.sampler)];
        var base_color = this.base_color_factor;
        base_color *= base_color_texture.Sample(base_color_sampler, disp_tex_coords).rgb;

        var normal_map_texture = textures[NonUniformResourceIndex(this.normal_map.texture)];
        var normal_map_sampler = samplers[NonUniformResourceIndex(this.normal_map.sampler)];
        var local_normal = (2.0 * normal_map_texture.Sample(normal_map_sampler, disp_tex_coords).xyz) - 1.0;
        local_normal = normalize(local_normal);
        var normal = normalize(
            local_normal.z * geometry_normal 
                + local_normal.x * tangent * this.normal_factor
                + local_normal.y * bitangent * this.normal_factor
        );

        var aorm_map_texture = textures[NonUniformResourceIndex(this.aorm_map.texture)];
        var aorm_map_sampler = samplers[NonUniformResourceIndex(this.aorm_map.sampler)];
        var aorm = float3(1.0, this.roughness_factor, this.metallic_factor);
        aorm *= aorm_map_texture.Sample(aorm_map_sampler, disp_tex_coords).rgb;

        FragmentProperties fragment;
        fragment.base_color = base_color;
        fragment.geometry_normal = geometry_normal;
        fragment.normal = normal;
        fragment.position = position;
        fragment.ambient_occlusion = aorm.r;

        CookTorranceBrdf brdf;
        brdf.alpha = aorm.g * aorm.g;
        brdf.metallic = aorm.b;
        brdf.ior = this.ior;
        
        return {
            fragment,
            brdf
        };
    }
};

struct LightUniforms {
    float3 ambient_color;
    float ambient_illuminance;
    uint point_count;
    uint directional_count;
};

[[vk::binding(0, 0)]]
Texture2D[] textures;
[[vk::binding(1, 0)]]
SamplerState[] samplers;

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target {
    let props = constants.materials[input.material_index].get_fragment_properties(
        constants.view.position,
        input.position,
        input.normal,
        input.tangent,
        input.handedness,
        input.tex_coords
    );
    var color = props.fragment.base_color 
        * props.fragment.ambient_occlusion
        * constants.light.ambient_color 
        * constants.light.ambient_illuminance;
    
    for (uint i = 0; i < constants.light.directional_count; i++) {
        color += calculate_radiance(
            constants.directional_lights[i], 
            props.brdf,
            props.fragment,
            constants.view.position
        );
    }
    
    for (uint i = 0; i < constants.light.point_count; i++) {
        color += calculate_radiance(
            constants.point_lights[i], 
            props.brdf,
            props.fragment,
            constants.view.position
        );
    }

    return float4(color, 1.0);
}