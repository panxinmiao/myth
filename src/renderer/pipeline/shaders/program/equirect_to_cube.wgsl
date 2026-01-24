@group(0) @binding(0)
var srcTex: texture_2d<f32>;

@group(0) @binding(1)
var s: sampler;

@group(0) @binding(2)
var dstTex: texture_storage_2d_array<rgba16float, write>;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

fn dirToUV(dir: vec3<f32>) -> vec2<f32> {
    let output = normalize(dir);
    var u = atan2(output.z, output.x) / TWO_PI + 0.5;
    var v = asin(clamp(output.y, -1.0, 1.0)) / PI + 0.5;
    return vec2<f32>(u, v);
}

fn getCubeDirection(face: u32, uv_normalized: vec2<f32>) -> vec3<f32> {
    let uv = 2.0 * uv_normalized - 1.0;
    switch (face) {
        case 0u: { return vec3f(1.0, -uv.y, -uv.x); }
        case 1u: { return vec3f(-1.0, -uv.y, uv.x); }
        case 2u: { return vec3f(uv.x, 1.0, uv.y); }
        case 3u: { return vec3f(uv.x, -1.0, -uv.y); }
        case 4u: { return vec3f(uv.x, -uv.y, 1.0); }
        case 5u: { return vec3f(-uv.x, -uv.y, -1.0); }
        default: { return vec3f(0.0); }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let face = id.z;
    let dims = textureDimensions(dstTex);
    
    if (id.x >= dims.x || id.y >= dims.y || face >= 6u) {
        return;
    }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(dims.xy);
    let dir = normalize(getCubeDirection(face, uv));
    let srcUV = dirToUV(dir);
    let color = textureSampleLevel(srcTex, s, srcUV, 0.0);
    textureStore(dstTex, vec2<u32>(id.xy), face, color);
}
