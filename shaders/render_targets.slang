[[vk::binding(0, 1)]]
Texture2D[] textures;
[[vk::binding(0, 1)]]
Texture2DArray[] texture_arrays;
[[vk::binding(0, 1)]]
TextureCube[] texture_cubes;
[[vk::binding(1, 1)]]
SamplerState[] samplers;
[[vk::binding(1, 1)]]
SamplerComparisonState[] compare_samplers;

public struct BindlessRenderTarget2D {
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
}

public struct BindlessShadowMapArray {
    uint texture;
    uint sampler;

    public Texture2DArray get_resource() {
        return texture_arrays[NonUniformResourceIndex(this.texture)];
    }

    public SamplerComparisonState get_sampler_resource() {
        return compare_samplers[NonUniformResourceIndex(this.sampler)];
    }

    public float sample(float3 coords, float compare) {
        return this.get_resource().SampleCmpLevelZero(this.get_sampler_resource(), coords, compare);
    }
}

public struct BindlessShadowMapCube {
    uint texture;
    uint sampler;

    public TextureCube get_resource() {
        return texture_cubes[NonUniformResourceIndex(this.texture)];
    }

    public SamplerComparisonState get_sampler_resource() {
        return compare_samplers[NonUniformResourceIndex(this.sampler)];
    }

    public float sample(float3 direction, float compare) {
        return this.get_resource().SampleCmpLevelZero(this.get_sampler_resource(), direction, compare);
    }
}