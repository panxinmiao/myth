@group(0) @binding(0)
var src_tex: texture_cube<f32>;

@group(0) @binding(1)
var src_sampler: sampler;

@group(0) @binding(2)
var dst_tex: texture_storage_2d_array<rgba16float, write>;

fn get_cube_direction(face: u32, uv_normalized: vec2<f32>) -> vec3<f32> {
    let uv = 2.0 * uv_normalized - 1.0;
    switch (face) {
        case 0u: { return vec3<f32>(1.0, -uv.y, -uv.x); }
        case 1u: { return vec3<f32>(-1.0, -uv.y, uv.x); }
        case 2u: { return vec3<f32>(uv.x, 1.0, uv.y); }
        case 3u: { return vec3<f32>(uv.x, -1.0, -uv.y); }
        case 4u: { return vec3<f32>(uv.x, -uv.y, 1.0); }
        case 5u: { return vec3<f32>(-uv.x, -uv.y, -1.0); }
        default: { return vec3<f32>(0.0); }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let face = id.z;
    let dims = textureDimensions(dst_tex);

    if (id.x >= dims.x || id.y >= dims.y || face >= 6u) {
        return;
    }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(dims.xy);
    let dir = normalize(get_cube_direction(face, uv));
    let color = textureSampleLevel(src_tex, src_sampler, dir, 0.0);
    let safe_color = clamp(color, vec4<f32>(0.0), vec4<f32>(65000.0));
    textureStore(dst_tex, vec2<u32>(id.xy), face, safe_color);
}