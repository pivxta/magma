import surface;
import brdf;
import cooktorrance;
import parallax;
import textures;

public struct Material {
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

    public Tuple<SurfaceSample, CookTorranceBrdf> sample(
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

            disp_tex_coords = parallax(
                this.displacement_map,
                this.displacement_factor,
                local_view_dir,
                tex_coords,
            );
        }

        var aorm = this.aorm_map.sample(disp_tex_coords).rgb;
        var base_color = this.base_color.sample(disp_tex_coords).rgb;
        let sampled_normal = this.normal_map.sample(disp_tex_coords).xyz;
        let local_normal = normalize((2.0 * sampled_normal) - 1.0);
        let normal = normalize(
            local_normal.z * geometry_normal 
                + local_normal.x * tangent * this.normal_factor
                + local_normal.y * bitangent * this.normal_factor
        );

        SurfaceSample sample;
        sample.base_color = base_color * this.base_color_factor;
        sample.geometry_normal = geometry_normal;
        sample.normal = normal;
        sample.position = position;
        sample.ambient_occlusion = aorm.r;

        CookTorranceBrdf brdf;
        brdf.metallic = aorm.b * this.metallic_factor;
        brdf.roughness = aorm.g * this.roughness_factor;
        brdf.alpha = brdf.roughness * brdf.roughness;
        brdf.ior = this.ior;
        
        return {sample, brdf};
    }
};