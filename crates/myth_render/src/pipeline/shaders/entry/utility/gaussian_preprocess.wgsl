// Gaussian Splatting Preprocess Shader
//
// Projects 3D Gaussians into 2D screen-space splats, evaluates view-dependent
// SH colour, and emits reverse-Z sort keys for back-to-front compositing.

const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C2 = array<f32, 5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32, 7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
};

struct GaussianCore {
    x: f32,
    y: f32,
    z: f32,
    opacity: u32,
    sh_idx: u32,
};

struct GaussianCovariance {
    cov: array<u32, 3>,
};

struct SplatGeometry {
    pos: vec2<f32>,
    v_0: u32,
    v_1: u32,
};

struct SplatAppearance {
    depth: f32,
    color_0: u32,
    color_1: u32,
    _pad: u32,
};

struct SortInfos {
    keys_size: atomic<u32>,
    padded_size: u32,
    passes: u32,
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
};

struct RenderSettings {
    gaussian_scaling: f32,
    max_sh_deg: u32,
    mip_splatting: u32,
    kernel_size: f32,
    scene_extent: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> gaussians: array<GaussianCore>;
@group(1) @binding(1)
var<storage, read> gaussian_covariances: array<GaussianCovariance>;
@group(1) @binding(2)
var<storage, read> sh_coefs: array<array<u32, 24>>;
@group(1) @binding(3)
var<storage, read_write> points_geom: array<SplatGeometry>;
@group(1) @binding(4)
var<storage, read_write> points_attr: array<SplatAppearance>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths: array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices: array<u32>;

@group(3) @binding(0)
var<uniform> render_settings: RenderSettings;

fn sh_coef(sh_idx: u32, c_idx: u32) -> vec3<f32> {
    let a = unpack2x16float(sh_coefs[sh_idx][(c_idx * 3u + 0u) / 2u])[(c_idx * 3u + 0u) % 2u];
    let b = unpack2x16float(sh_coefs[sh_idx][(c_idx * 3u + 1u) / 2u])[(c_idx * 3u + 1u) % 2u];
    let c = unpack2x16float(sh_coefs[sh_idx][(c_idx * 3u + 2u) / 2u])[(c_idx * 3u + 2u) % 2u];
    return vec3<f32>(a, b, c);
}

fn evaluate_sh(dir: vec3<f32>, sh_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(sh_idx, 0u);

    if sh_deg > 0u {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;
        result += -SH_C1 * y * sh_coef(sh_idx, 1u)
                + SH_C1 * z * sh_coef(sh_idx, 2u)
                - SH_C1 * x * sh_coef(sh_idx, 3u);

        if sh_deg > 1u {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let yz = y * z;
            let xz = x * z;
            result += SH_C2[0] * xy * sh_coef(sh_idx, 4u)
                    + SH_C2[1] * yz * sh_coef(sh_idx, 5u)
                    + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(sh_idx, 6u)
                    + SH_C2[3] * xz * sh_coef(sh_idx, 7u)
                    + SH_C2[4] * (xx - yy) * sh_coef(sh_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(sh_idx, 9u)
                        + SH_C3[1] * xy * z * sh_coef(sh_idx, 10u)
                        + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(sh_idx, 11u)
                        + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(sh_idx, 12u)
                        + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(sh_idx, 13u)
                        + SH_C3[5] * z * (xx - yy) * sh_coef(sh_idx, 14u)
                        + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(sh_idx, 15u);
            }
        }
    }
    result += 0.5;
    return result;
}

fn cov_coefs(v_idx: u32) -> array<f32, 6> {
    let a = unpack2x16float(gaussian_covariances[v_idx].cov[0]);
    let b = unpack2x16float(gaussian_covariances[v_idx].cov[1]);
    let c = unpack2x16float(gaussian_covariances[v_idx].cov[2]);
    return array<f32, 6>(a.x, a.y, b.x, b.y, c.x, c.y);
}

fn safe_normalize(v: vec2<f32>) -> vec2<f32> {
    let len_sq = dot(v, v);
    if len_sq > 1e-12 {
        return v * inverseSqrt(len_sq);
    }
    return vec2<f32>(1.0, 0.0);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&gaussians) {
        return;
    }

    let focal = camera.focal;
    let viewport = camera.viewport;
    let vertex = gaussians[idx];
    let a = unpack2x16float(vertex.opacity);
    let xyz = vec3<f32>(vertex.x, -vertex.y, -vertex.z);
    var opacity = a.x;

    let camspace = camera.view * vec4<f32>(xyz, 1.0);
    let pos2d = camera.proj * camspace;
    let bounds = 1.2 * pos2d.w;
    let center_depth = pos2d.z / pos2d.w;

    // Frustum culling
    if center_depth <= 0.0 || center_depth > 1.0
        || pos2d.x < -bounds || pos2d.x > bounds
        || pos2d.y < -bounds || pos2d.y > bounds {
        return;
    }

    let raw_c = cov_coefs(idx);
    let cov_sparse = array<f32, 6>(
        raw_c[0], -raw_c[1], -raw_c[2], // c00, -c01, -c02
        raw_c[3], raw_c[4], raw_c[5]    // c11, c12, c22
    );
    let scaling = render_settings.gaussian_scaling;

    let Vrk = mat3x3<f32>(
        cov_sparse[0], cov_sparse[1], cov_sparse[2],
        cov_sparse[1], cov_sparse[3], cov_sparse[4],
        cov_sparse[2], cov_sparse[4], cov_sparse[5]
    ) * scaling * scaling;

    let J = mat3x3<f32>(
        focal.x / camspace.z, 0.0, -(focal.x * camspace.x) / (camspace.z * camspace.z),
        0.0, focal.y / camspace.z, -(focal.y * camspace.y) / (camspace.z * camspace.z),
        0.0, 0.0, 0.0
    );

    let W = transpose(mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    ));
    let T = W * J;
    let cov = transpose(T) * Vrk * T;

    let kernel_size = render_settings.kernel_size;
    if render_settings.mip_splatting != 0u {
        let det_0 = max(1e-6, cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);
        let det_1 = max(1e-6, (cov[0][0] + kernel_size) * (cov[1][1] + kernel_size) - cov[0][1] * cov[0][1]);
        var coef = sqrt(det_0 / (det_1 + 1e-6) + 1e-6);
        if det_0 <= 1e-6 || det_1 <= 1e-6 {
            coef = 0.0;
        }
        opacity *= coef;
    }

    let diagonal1 = cov[0][0] + kernel_size;
    let offDiagonal = cov[0][1];
    let diagonal2 = cov[1][1] + kernel_size;

    let mid = 0.5 * (diagonal1 + diagonal2);
    let radius = length(vec2<f32>((diagonal1 - diagonal2) / 2.0, offDiagonal));
    let lambda1 = mid + radius;
    let lambda2 = max(mid - radius, 0.1);

    let diagonalVector = safe_normalize(vec2<f32>(offDiagonal, lambda1 - diagonal1));
    let v1 = sqrt(2.0 * lambda1) * diagonalVector;
    let v2 = sqrt(2.0 * lambda2) * vec2<f32>(diagonalVector.y, -diagonalVector.x);

    let center_ndc = pos2d.xy / pos2d.w;

    let camera_pos = camera.view_inv[3].xyz;
    let dir = normalize(xyz - camera_pos);
    let sh_idx = min(vertex.sh_idx, max(arrayLength(&sh_coefs), 1u) - 1u);

    let dir_colmap = vec3<f32>(dir.x, -dir.y, -dir.z);

    let color = vec4<f32>(
        max(vec3<f32>(0.0), evaluate_sh(dir_colmap, sh_idx, render_settings.max_sh_deg)),
        opacity
    );

    let store_idx = atomicAdd(&sort_infos.keys_size, 1u);
    let v = vec4<f32>(v1 / viewport, v2 / viewport);
    points_geom[store_idx] = SplatGeometry(
        center_ndc,
        pack2x16float(v.xy),
        pack2x16float(v.zw),
    );
    points_attr[store_idx] = SplatAppearance(
        center_depth,
        pack2x16float(color.rg),
        pack2x16float(color.ba),
        0u,
    );

    sort_depths[store_idx] = bitcast<u32>(center_depth);
    sort_indices[store_idx] = store_idx;

    let keys_per_wg = 256u * 15u;
    if (store_idx % keys_per_wg) == 0u {
        atomicAdd(&sort_infos.dispatch_x, 1u);
    }
}
