[[vk::binding(0, 0)]]
Texture2D[] textures;
[[vk::binding(0, 0)]]
Texture2DArray[] texture_arrays;
[[vk::binding(1, 0)]]
SamplerState[] samplers;
[[vk::binding(1, 0)]]
SamplerComparisonState[] compare_samplers;

public struct BindlessTexture2D {
    uint texture;
    uint sampler;

    public Texture2D get_resource() {
        return textures[NonUniformResourceIndex(this.texture)];
    }

    public SamplerState get_sampler_resource() {
        return samplers[NonUniformResourceIndex(this.sampler)];
    }

    public float4 sample(float2 coords) {
        return this.get_resource().Sample(this.get_sampler_resource(), coords);
    }

    public float4 sample_grad(float2 coords, float2 dx, float2 dy) {
        return this.get_resource().SampleGrad(this.get_sampler_resource(), coords, dx, dy);
    }
}
