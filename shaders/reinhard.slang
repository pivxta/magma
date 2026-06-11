float3 reinhard(float3 linear_color) {
    return linear_color / (1.0 + linear_color);
}

