import brdf;
import surface;
import consts;

public struct CookTorranceBrdf: Brdf {
    public float roughness;
    public float alpha;
    public float metallic;
    public float ior;

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

    float3 fresnel_f0(float3 base_color) {
        let dieletric = (this.ior - 1.0) / (this.ior + 1.0);
        return lerp(float3(dieletric * dieletric), base_color, this.metallic);
    }

    float3 fresnel_schlick_roughness(float3 f0, float wo_dot_n) {
        let f90 = max(float3(1.0 - this.roughness), f0);
        return f0 + (f90 - f0) * pow(1.0 - wo_dot_n, 5.0);
    }

    float3 fresnel_schlick(float3 f0, float wo_dot_wh) {
        return f0 + (1.0 - f0) * pow(1.0 - wo_dot_wh, 5.0);
    }

    public float3 calculate(
        SurfaceSample sample,
        float3 omega_i,
        float3 omega_o
    ) {
        float3 omega_h = normalize(omega_i + omega_o);
        float wi_dot_n = max(dot(omega_i, sample.normal), 0.000001);
        float wo_dot_n = max(dot(omega_o, sample.normal), 0.000001);
        float wh_dot_n = max(dot(omega_h, sample.normal), 0.0);
        float wo_dot_wh = max(dot(omega_o, omega_h), 0.0);

        float3 f = this.fresnel_schlick(this.fresnel_f0(sample.base_color), wo_dot_wh);
        float3 v = this.visibility_smith(wi_dot_n, wo_dot_n);
        float d = this.microfacet_ggx(wh_dot_n);
        float3 specular = f * d * v;
        float3 diffuse = (1.0 - this.metallic) * (1.0 - f) * sample.base_color / PI;
        return specular + diffuse;
    }
    
    float2 env_brdf_approx(float wo_dot_n) {
        let c0 = float4(-1.0, -0.0275, -0.572, 0.022);
        let c1 = float4(1.0, 0.0425, 1.04, -0.04);
        let r = this.roughness * c0 + c1;
        let a004 = min(r.x * r.x, exp2(-9.28 * wo_dot_n)) * r.x + r.y;
        return float2(-1.04, 1.04) * a004 + r.zw;
    }

    public float3 calculate_ambient(
        SurfaceSample sample,
        float3 omega_o,
        float3 ambient_radiance
    ) {
        let wo_dot_n = max(dot(omega_o, sample.normal), 0.0001);
        let f0 = this.fresnel_f0(sample.base_color);
        let fresnel = this.fresnel_schlick_roughness(f0, wo_dot_n);

        let diffuse = (1.0 - fresnel) 
            * (1.0 - this.metallic) 
            * sample.base_color 
            * ambient_radiance;
        let env = this.env_brdf_approx(wo_dot_n);
        let specular = ambient_radiance * (f0 * env.x + env.y);

        return (diffuse + specular) * sample.ambient_occlusion;
    }
}

