// Gaussian Splatting — Radix Sort: Histogram Pass
//
// Counts per-digit frequencies across work-groups for the current
// 8-bit radix pass. One dispatch processes ALL keys.

const WG_SIZE: u32 = 256u;
const KEYS_PER_THREAD: u32 = 15u;
const KEYS_PER_WG: u32 = 3840u; // WG_SIZE * KEYS_PER_THREAD

struct SortInfos {
    keys_size: atomic<u32>,
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
};

@group(0) @binding(0) var<storage, read_write> sort_infos: SortInfos;
@group(0) @binding(1) var<storage, read_write> sort_depths: array<u32>;
@group(0) @binding(2) var<storage, read_write> sort_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> sort_dispatch: array<u32>;

@group(1) @binding(0) var<storage, read_write> assist_a: array<u32>;
@group(1) @binding(1) var<storage, read_write> assist_b: array<u32>;
@group(1) @binding(2) var<storage, read_write> histograms: array<atomic<u32>>;

fn extract_digit(key: u32, radix_pass: u32) -> u32 {
    return (key >> (radix_pass * 8u)) & 0xFFu;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let num_keys = atomicLoad(&sort_infos.keys_size);
    if num_keys == 0u {
        return;
    }
    let num_wgs = (num_keys + KEYS_PER_WG - 1u) / KEYS_PER_WG;
    let radix_pass = sort_infos.passes;
    let wg = gid.x / WG_SIZE;

    for (var i = 0u; i < KEYS_PER_THREAD; i++) {
        let data_idx = wg * KEYS_PER_WG + i * WG_SIZE + lid.x;
        if data_idx < num_keys {
            let key = select(sort_depths[sort_indices[data_idx]], sort_depths[data_idx], bool(sort_infos.even_pass));
            let digit = extract_digit(key, radix_pass);
            atomicAdd(&histograms[radix_pass * num_wgs * 256u + digit * num_wgs + wg], 1u);
        }
    }
}
