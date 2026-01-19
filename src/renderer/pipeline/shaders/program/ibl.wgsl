struct Params {
  roughness: f32,
  resolution: f32,
};

@group(0) @binding(0)
var srcTex: texture_cube<f32>;

@group(0) @binding(1)
var s: sampler;

@group(0) @binding(2)
var<uniform> params: Params;

@group(1) @binding(0)
var destTex: texture_storage_2d_array<rgba16float, write>;

const PI: f32 = 3.141592653589793;
const SAMPLE_COUNT: u32 = 4096;

//compute the direction vector for a given cube map face and texel coordinate
fn getCubeDirection(face: u32, uv_: vec2<f32>) -> vec3<f32> {
  let uv = 2.0 * uv_ - 1.0;
  switch (face) {
    case 0u: {
        return vec3f(1.0, -uv.y, -uv.x);  // +X
    }
    case 1u: {
        return vec3f(-1.0, -uv.y, uv.x);  // -X
    }
    case 2u: {
        return vec3f(uv.x, 1.0, uv.y);    // +Y
    }
    case 3u: {
        return vec3f(uv.x, -1.0, -uv.y);  // -Y
    }
    case 4u: {
        return vec3f(uv.x, -uv.y, 1.0);   // +Z
    }
    case 5u: {
        return vec3f(-uv.x, -uv.y, -1.0); // -Z
    }
    default: {
        return vec3f(0.0);
    }
  }
}

// Hammersley sequence
fn hammersley(i: u32, N: u32) -> vec2<f32> {
    var bits = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    let radicalInverse = f32(bits) * 2.3283064365386963e-10;
    return vec2f(f32(i)/f32(N), radicalInverse);
}

// GGX importance sampling
fn importanceSampleGGX(xi: vec2<f32>, N: vec3<f32>, roughness: f32) -> vec3<f32> {
  let a = roughness * roughness;
  
  let phi = 2.0 * PI * xi.x;
  let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a*a - 1.0) * xi.y));
  let sinTheta = sqrt(1.0 - cosTheta * cosTheta);
  
  let H = vec3f(
    cos(phi) * sinTheta,
    sin(phi) * sinTheta,
    cosTheta
  );
  
  let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(N.z) < 0.999);
  let tangent = normalize(cross(up, N));
  let bitangent = cross(N, tangent);
  
  let sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
  return normalize(sampleVec);
}

// GGX normal distribution function
fn distributionGGX(NdotH: f32, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (PI * denom * denom);
}

// todo: override workgroup_size
const workgroup_size: u32 = 8u;

@compute
@workgroup_size(workgroup_size, workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let face = id.z;

  let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(params.resolution);
  
  let N = normalize(getCubeDirection(face, uv));
  let R = N;
  let V = R;
  
  var prefilteredColor = vec3<f32>(0.0);
  var totalWeight = 0.0;
  
  for (var i: u32 = 0u; i < SAMPLE_COUNT; i = i + 1u) {
    let xi = hammersley(i, SAMPLE_COUNT);
    let H = importanceSampleGGX(xi, N, params.roughness);
    let L = normalize(2.0 * dot(V, H) * H - V);
    
    // let NdotL = max(dot(N, L), 0.0);
    // let NdotL = clamp(dot(N, L), 0.0, 1.0);
    let NdotL = saturate(dot(N, L));
    if (NdotL > 0.0) {
      let NdotH = clamp(dot(N, H), 0.0, 1.0);
      let HdotV = clamp(dot(H, V), 0.0, 1.0);
      
      let D = distributionGGX(NdotH, params.roughness);
      let pdf = D * NdotH / (4.0 * HdotV) + 0.0001;
      let texSize = textureDimensions(srcTex).x;
      let saTexel = 4.0 * PI / (6.0 * f32(texSize) * f32(texSize));
      let saSample = 1.0 / (f32(SAMPLE_COUNT) * pdf + 0.0001);
      let mipLevel = select(0.5 * log2(saSample / saTexel), 0.0, params.roughness == 0.0);

      prefilteredColor += textureSampleLevel(srcTex, s, L, mipLevel).rgb * NdotL;
      totalWeight += NdotL;
    }
  }

  prefilteredColor = prefilteredColor / totalWeight;
  textureStore(destTex, vec2<u32>(id.xy), face, vec4<f32>(prefilteredColor, 1.0));
}