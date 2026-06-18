import consts;
import render_targets;

interface Light {
    bool should_cull(float3 position);
    float get_shadow_factor(float3 position, float4x4 world_to_view);
    float3 get_incident_radiance(float3 position);
    float3 get_omega_i(float3 position);
};

static const uint MAX_CASCADES = 4;

struct DirectionalLight: Light {
    float3 direction;
    float3 color;
    float illuminance;

    bool shadows_enabled;
    uint cascade_count;
    float[MAX_CASCADES] cascade_depths;
    float4x4[MAX_CASCADES] cascade_world_to_clip;
    BindlessArrayShadowMap shadow_map;

    bool should_cull(float3 position) {
        return false;
    }

    float get_shadow_factor(float3 position, float4x4 world_to_view) {
        if (!this.shadows_enabled || this.cascade_count == 0) {
            return 1.0;
        }

        let position_view_space = mul(world_to_view, float4(position, 1.0)).xyz;
        uint cascade = this.cascade_count - 1;
        for (uint i = 0; i < this.cascade_count - 1; i++) {
            if (position_view_space.z < this.cascade_depths[i]) {
                cascade = i;
                break;
            }
        }

        // What the hell is this loop is doing here? you might be asking yourself
        // Ask my freaking RX580s stupid windows GPU driver that cant index a fucking
        // matrix pointer correctly
        float4x4 world_to_clip = this.cascade_world_to_clip[0];
        [ForceUnroll]
        for (uint i = 0; i < MAX_CASCADES; i++) {
            if (cascade == i) {
                world_to_clip = this.cascade_world_to_clip[i];
            }
        }

        var shadow_clip_position = mul(world_to_clip, float4(position, 1.0));
        shadow_clip_position /= shadow_clip_position.w;
        if (shadow_clip_position.x < -1.0 
            || shadow_clip_position.x > 1.0
            || shadow_clip_position.y < -1.0
            || shadow_clip_position.y > 1.0) {
            return 1.0;
        }

        let shadow_map_coords = float2(0.5 * shadow_clip_position.xy + 0.5);
        return this.shadow_map.sample(cascade, shadow_map_coords, shadow_clip_position.z);
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
    BindlessCubeShadowMap shadow_map;

    bool should_cull(float3 position) {
        let fragment_to_light = this.position - position;
        return dot(fragment_to_light, fragment_to_light) > this.radius * this.radius;
    }

    float get_shadow_factor(float3 position, float4x4 world_to_view) {
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
