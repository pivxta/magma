public struct AgxLook {
    public float saturation;
    public float3 slope;
    public float3 offset;
    public float3 power;
};

public float3 agx(AgxLook look, float3 linear_color) {
    var value = mul(agx_matrix, linear_color);
    value = clamp(log2(max(value, 1e-6f)), min_ev, max_ev);
    value = (value - min_ev) / (max_ev - min_ev);
    value = agx_default_contrast(value);

    const float3 luma_weights = float3(0.2126, 0.7152, 0.0722);
    const float luma = dot(value, luma_weights);
    
    value = pow(value * look.slope + look.offset, look.power);
    value = luma + look.saturation * (value - luma);
    value = mul(agx_inverse_matrix, value);
    return pow(value, 2.2);
}

float3 agx_default_contrast(float3 x) {
    let x2 = x * x;
    let x4 = x2 * x2;
    return + 15.5 * x4 * x2
         - 40.14 * x4 * x
         + 31.96 * x4
         - 6.868 * x2 * x
         + 0.4298 * x2
         + 0.1191 * x
         - 0.00232;
}

static const float3x3 agx_matrix = float3x3(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772, 0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104
);

static const float3x3 agx_inverse_matrix = float3x3(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433, 1.15107367264116
);

static const float min_ev = -12.47393f;
static const float max_ev = 4.026069f;
