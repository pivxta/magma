[[vk::binding(0, 0)]]
Texture2D[] textures;
[[vk::binding(1, 0)]]
SamplerState[] samplers;

public struct TextureIndices {
    uint texture;
    uint sampler;

    public Texture2D get_texture() {
        return textures[NonUniformResourceIndex(this.texture)];
    }

    public SamplerState get_sampler() {
        return samplers[NonUniformResourceIndex(this.sampler)];
    }

    public float4 sample(float2 coords) {
        return this.get_texture().Sample(this.get_sampler(), coords);
    }

    public float4 sample_grad(float2 coords, float2 dx, float2 dy) {
        return this.get_texture().SampleGrad(this.get_sampler(), coords, dx, dy);
    }
}
