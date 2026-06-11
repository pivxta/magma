import surface;
import light;

public interface Brdf {
    public float3 calculate(
        SurfaceSample sample,
        float3 omega_i,
        float3 omega_o
    );
    public float3 calculate_ambient(
        SurfaceSample sample,
        float3 omega_o,
        float3 ambient_radiance
    );
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

public float3 calculate_radiance<B: Brdf, L: Light>(
    L light,
    B brdf,
    SurfaceSample sample,
    float3 view_position,
) {
    if (light.should_cull(sample.position)) {
        return float3(0.0);
    }

    let omega_i = light.get_omega_i(sample.position);
    let omega_o = normalize(view_position - sample.position);
    let shadow_mask = shadow_terminator_mask(omega_i, sample.normal, sample.geometry_normal);
    return brdf.calculate(sample, omega_i, omega_o)
            * light.get_incident_radiance(sample.position)
            * max(dot(omega_i, sample.normal), 0.0)
            * shadow_mask;
}