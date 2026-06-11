import consts;

interface Light {
    bool should_cull(float3 position);
    float3 get_incident_radiance(float3 position);
    float3 get_omega_i(float3 position);
};

struct DirectionalLight: Light {
    float3 direction;
    float padding;
    float3 color;
    float illuminance;

    bool should_cull(float3 position) {
        return false;
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

    bool should_cull(float3 position) {
        let fragment_to_light = this.position - position;
        return dot(fragment_to_light, fragment_to_light) > this.radius * this.radius;
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
