@group(0) @binding(0)
var lutTex: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.141592653589793;
const SAMPLE_COUNT: u32 = 1024u;

// Hammersley Sequence
fn hammersley(i: u32, N: u32) -> vec2<f32> {
    var bits = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    let radicalInverse = f32(bits) * 2.3283064365386963e-10;
    return vec2f(f32(i)/f32(N), radicalInverse);
}

// GGX Importance Sampling
fn importanceSampleGGX(xi: vec2<f32>, roughness: f32, N: vec3<f32>) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a*a - 1.0) * xi.y));
    let sinTheta = sqrt(1.0 - cosTheta*cosTheta);
    let H = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    
    // Tangent Space
    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(N.z) < 0.999);
    let tangent = normalize(cross(up, N));
    let bitangent = cross(N, tangent);
    
    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let k = (roughness * roughness) / 2.0;
    let nom   = NdotV;
    let denom = NdotV * (1.0 - k) + k;
    return nom / denom;
}

fn geometrySmith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let ggx2  = geometrySchlickGGX(NdotV, roughness);
    let ggx1  = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

fn integrateBRDF(NdotV: f32, roughness: f32) -> vec2<f32> {
    var V = vec3f(sqrt(1.0 - NdotV*NdotV), 0.0, NdotV);
    var A = 0.0;
    var B = 0.0;
    let N = vec3f(0.0, 0.0, 1.0);

    for(var i = 0u; i < SAMPLE_COUNT; i++) {
        let xi = hammersley(i, SAMPLE_COUNT);
        let H  = importanceSampleGGX(xi, roughness, N);
        let L  = normalize(2.0 * dot(V, H) * H - V);

        let NdotL = saturate(L.z);
        let NdotH = saturate(H.z);
        let VdotH = saturate(dot(V, H));

        if(NdotL > 0.0) {
            let G = geometrySmith(NdotV, NdotL, roughness);
            let G_Vis = (G * VdotH) / (NdotH * NdotV);
            let Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    return vec2f(A, B) / f32(SAMPLE_COUNT);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2f(textureDimensions(lutTex));
    let uv = vec2f(id.xy) / size;
    // x: NdotV, y: Roughness
    // 注意：我们要防止 uv.y (roughness) 为 0 导致的计算问题，通常加一个小 epsilon
    let roughness = max(uv.y, 0.001);
    let integratedBRDF = integrateBRDF(uv.x, roughness);
    textureStore(lutTex, id.xy, vec4f(integratedBRDF, 0.0, 1.0));
}