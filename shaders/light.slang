import consts;
import render_targets;

interface Light {
    bool should_cull(float3 position);
    float get_shadow_factor(float3 position);
    float3 get_incident_radiance(float3 position);
    float3 get_omega_i(float3 position);
};

struct DirectionalLight: Light {
    float3 direction;
    float padding;
    float3 color;
    float illuminance;
    bool shadows_enabled;
    float4x4 world_to_shadow_clip;
    RenderTargetIndices shadow_map;

    bool should_cull(float3 position) {
        return false;
    }

    float get_shadow_factor(float3 position) {
        if (!this.shadows_enabled) {
            return 1.0;
        }

        var shadow_clip_position = mul(this.world_to_shadow_clip, float4(position, 1.0));
        shadow_clip_position /= shadow_clip_position.w;
        if (shadow_clip_position.x < -1.0 
            || shadow_clip_position.x > 1.0
            || shadow_clip_position.y < -1.0
            || shadow_clip_position.y > 1.0) {
            return 1.0;
        }

        let shadow_map_coords = 0.5 * shadow_clip_position.xy + 0.5;
        return this.shadow_map.sample_cmp(shadow_map_coords, shadow_clip_position.z);
    }

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
    bool shadows_enabled;
    RenderTargetIndices shadow_map;

    bool should_cull(float3 position) {
        let fragment_to_light = this.position - position;
        return dot(fragment_to_light, fragment_to_light) > this.radius * this.radius;
    }

    float get_shadow_factor(float3 position) {
        return 1.0;
    }

    float3 get_incident_radiance(float3 position) {
        let fragment_to_light = this.position - position;
        let distance_sq = dot(fragment_to_light, fragment_to_light);
        var fade_out = saturate(1.0 - pow(distance_sq / (this.radius * this.radius), 2.0));
        fade_out *= fade_out;
        return (fade_out * this.color * this.intensity) / max(4.0 * PI * distance_sq, 0.0001);
    }

    float3 get_omega_i(float3 position) {
        return normalize(this.position - position);
    }
};
