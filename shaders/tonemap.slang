import agx;
import reinhard;
import render_targets;

struct BlitVertex {
    float4 clip_position: SV_Position;
    float2 tex_coords;
}

[shader("vertex")]
BlitVertex vertex_main(uint vertex_id: SV_VertexID) {
    float2 uv = float2(vertex_id & 2, (vertex_id << 1) & 2);
    BlitVertex output;
    output.tex_coords = uv;
    output.clip_position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

enum Tonemap {
    None = 0,
    Reinhard = 1,
    Agx = 2
};

struct TonemapConstants {
    RenderTargetIndices input;
    Tonemap tonemap;
    float exposure;
    AgxLook agx_look;
}

[[vk::push_constant]]
TonemapConstants constants;

[shader("fragment")]
float4 fragment_main(BlitVertex input) : SV_Target {
    let input_color = constants.exposure * constants.input.sample(input.tex_coords).rgb;
    var output_color = input_color;
    if (constants.tonemap == Tonemap::Reinhard) {
        output_color = reinhard(input_color);
    } else if (constants.tonemap == Tonemap::Agx) {
        output_color = agx(constants.agx_look, input_color);
    }
    return float4(output_color, 1.0);
}