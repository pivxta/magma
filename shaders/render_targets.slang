[[vk::binding(0, 1)]]
Texture2D[] textures;
[[vk::binding(0, 1)]]
Texture2DArray[] texture_arrays;
[[vk::binding(0, 1)]]
TextureCubeArray[] texture_cube_arrays;
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

public struct BindlessArrayShadowMapAtlas {
    uint texture;
    uint sampler;

    public Texture2DArray get_resource() {
        return texture_arrays[NonUniformResourceIndex(this.texture)];
    }

    public SamplerComparisonState get_sampler_resource() {
        return compare_samplers[NonUniformResourceIndex(this.sampler)];
    }

    public float sample(uint index, float2 coords, float compare) {
        return this.get_resource().SampleCmpLevelZero(
            this.get_sampler_resource(), 
            float3(coords, float(index)), 
            compare
        );
    }
}

public struct BindlessCubeShadowMapAtlas {
    uint texture;
    uint sampler;

    public TextureCubeArray get_resource() {
        return texture_cube_arrays[NonUniformResourceIndex(this.texture)];
    }

    public SamplerComparisonState get_sampler_resource() {
        return compare_samplers[NonUniformResourceIndex(this.sampler)];
    }

    public float sample(uint index, float3 direction, float compare) {
        return this.get_resource().SampleCmpLevelZero(
            this.get_sampler_resource(), 
            float4(direction, float(index)), 
            compare
        );
    }
}

public struct BindlessArrayShadowMap {
    BindlessArrayShadowMapAtlas atlas;
    uint32_t layer;

    public float sample(uint layer, float2 coords, float compare) {
        return this.atlas.sample(this.layer + layer, coords, compare);
    }
}

public struct BindlessCubeShadowMap {
    BindlessCubeShadowMapAtlas atlas;
    uint32_t cube;

    public float sample(float3 direction, float compare) {
        return this.atlas.sample(cube, direction, compare);
    }
}