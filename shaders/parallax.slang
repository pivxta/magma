import textures;

float2 parallax(
    TextureIndices displacement_texture,
    float displacement_factor,
    float3 local_view_dir,
    float2 tex_coords,
    float min_layers = 4.0,
    float max_layers = 16.0
) {
    let num_layers = lerp(max_layers, min_layers, abs(local_view_dir.z));
    let layer_depth = 1.0 / num_layers;
    var current_layer_depth = 0.0;

    let dx = ddx(tex_coords);
    let dy = ddy(tex_coords);

    let z = max(local_view_dir.z, 0.0);
    let delta_tex_coords = (local_view_dir.xy * displacement_factor / z) / num_layers;
    var current_tex_coords = tex_coords;
    var current_depth = 1.0 - displacement_texture.sample_grad(current_tex_coords, dx, dy).r;

    [loop]
    while (current_layer_depth < current_depth) {
        current_tex_coords -= delta_tex_coords;
        current_layer_depth += layer_depth;
        current_depth = 1.0 - displacement_texture.sample_grad(current_tex_coords, dx, dy).r;
    }

    let prev_tex_coords = current_tex_coords + delta_tex_coords;
    let after_depth  = current_depth - current_layer_depth;
    let before_depth = 
        (1.0 - displacement_texture.sample_grad(prev_tex_coords, dx, dy).r)
            - current_layer_depth 
            + layer_depth;

    let weight = after_depth / (after_depth - before_depth);
    return lerp(current_tex_coords, prev_tex_coords, weight);
}