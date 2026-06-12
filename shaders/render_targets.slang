[[vk::binding(0, 1)]]
Texture2D[] render_targets;
[[vk::binding(1, 1)]]
SamplerState[] render_target_samplers;

public struct RenderTargetIndices {
    uint texture;
    uint sampler;

    public Texture2D get_texture() {
        return render_targets[NonUniformResourceIndex(this.texture)];
    }

    public SamplerState get_sampler() {
        return render_target_samplers[NonUniformResourceIndex(this.sampler)];
    }

    public float4 sample(float2 coords) {
        return this.get_texture().Sample(this.get_sampler(), coords);
    }
}
