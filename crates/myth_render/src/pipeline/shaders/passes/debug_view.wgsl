// Debug View — visualise intermediate RDG textures.
//
// A full-screen post-process pass that remaps arbitrary texture formats
// into a displayable [0, 1] RGB range.  The `view_mode` uniform selects
// the mapping strategy so a single pipeline can handle depth, normals,
// single-channel occlusion, and standard colour buffers alike.

{$ include 'full_screen_vertex.wgsl' $}

struct DebugUniforms {
    // 0: RGB pass-through
    // 1: Single-channel R → grayscale (e.g. SSAO)
    // 2: Signed vector [-1,1] → [0,1] (e.g. normals, velocity)
    // 3: Linear depth visualisation
    view_mode: u32,
    // _pad: vec3<u32>,
};

// Group 0 (static): sampler + uniforms — owned by the Feature.
@group(0) @binding(0) var debug_sampler: sampler;
@group(0) @binding(1) var<uniform> uniforms: DebugUniforms;

// Group 1 (transient): source texture — rebuilt each frame.
@group(1) @binding(0) var debug_texture: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSampleLevel(debug_texture, debug_sampler, in.uv, 0.0);

    switch uniforms.view_mode {
        // Single-channel grayscale (SSAO, shadow mask, etc.)
        case 1u: {
            return vec4<f32>(tex_color.rrr, 1.0);
        }
        // Signed → unsigned mapping (normals, motion vectors)
        case 2u: {
            return vec4<f32>(tex_color.rgb * 0.5 + 0.5, 1.0);
        }
        // Reverse-Z depth — simple pow ramp for perceptual contrast
        case 3u: {
            let z = clamp(tex_color.r, 0.0, 1.0);
            let linear = pow(1.0 - z, 64.0);
            return vec4<f32>(vec3<f32>(linear), 1.0);
        }
        // Default: colour pass-through
        default: {
            return vec4<f32>(tex_color.rgb, 1.0);
        }
    }
}
