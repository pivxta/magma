[[vk::binding(0, 1)]]
Texture2D[] targets;
[[vk::binding(1, 1)]]
SamplerState[] target_samplers;
[[vk::binding(2, 1)]]
SamplerComparisonState[] target_cmp_samplers;

public struct RenderTargetIndices {
    uint texture;
    uint sampler;

    public Texture2D get_texture() {
        return targets[NonUniformResourceIndex(this.texture)];
    }

    public SamplerState get_sampler() {
        return target_samplers[NonUniformResourceIndex(this.sampler)];
    }

    public SamplerComparisonState get_cmp_sampler() {
        return target_cmp_samplers[NonUniformResourceIndex(this.sampler)];
    }

    public float4 sample(float2 coords) {
        return this.get_texture().Sample(this.get_sampler(), coords);
    }

    public float sample_cmp(float2 coords, float compare) {
        return this.get_texture().SampleCmpLevelZero(this.get_cmp_sampler(), coords, compare);
    }
}
