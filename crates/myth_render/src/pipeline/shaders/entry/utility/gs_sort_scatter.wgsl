// Gaussian Splatting — Radix Sort: Scatter Pass
//
// Scatters keys and values (sort indices) into their globally sorted
// positions using the prefix sums from the histogram + prefix passes.

const WG_SIZE: u32 = 256u;
const KEYS_PER_THREAD: u32 = 15u;
const KEYS_PER_WG: u32 = 3840u;

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

var<workgroup> s_local_histo: array<atomic<u32>, 256>;
var<workgroup> s_prefix: array<u32, 256>;

fn extract_digit(key: u32, radix_pass: u32) -> u32 {
    return (key >> (radix_pass * 8u)) & 0xFFu;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let num_keys = atomicLoad(&sort_infos.keys_size);
    if num_keys == 0u {
        return;
    }
    let num_wgs = (num_keys + KEYS_PER_WG - 1u) / KEYS_PER_WG;
    let radix_pass = sort_infos.passes;
    let wg = wid.x;
    let global_offset = radix_pass * num_wgs * 256u;

    // Clear local histogram
    atomicStore(&s_local_histo[lid.x], 0u);
    workgroupBarrier();

    // Build local histogram for this work-group's keys
    for (var i = 0u; i < KEYS_PER_THREAD; i++) {
        let data_idx = wg * KEYS_PER_WG + i * WG_SIZE + lid.x;
        if data_idx < num_keys {
            let key = select(sort_depths[sort_indices[data_idx]], sort_depths[data_idx], bool(sort_infos.even_pass));
            let digit = extract_digit(key, radix_pass);
            atomicAdd(&s_local_histo[digit], 1u);
        }
    }
    workgroupBarrier();

    // Compute local prefix sum
    s_prefix[lid.x] = atomicLoad(&s_local_histo[lid.x]);
    workgroupBarrier();

    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var temp = 0u;
        if lid.x >= stride {
            temp = s_prefix[lid.x - stride];
        }
        workgroupBarrier();
        s_prefix[lid.x] += temp;
        workgroupBarrier();
    }

    // Clear for per-digit scatter counting
    atomicStore(&s_local_histo[lid.x], 0u);
    workgroupBarrier();

    // Scatter keys + values to global positions
    for (var i = 0u; i < KEYS_PER_THREAD; i++) {
        let data_idx = wg * KEYS_PER_WG + i * WG_SIZE + lid.x;
        if data_idx < num_keys {
            let key = select(sort_depths[sort_indices[data_idx]], sort_depths[data_idx], bool(sort_infos.even_pass));
            let value = select(sort_indices[data_idx], data_idx, bool(sort_infos.even_pass));
            let digit = extract_digit(key, radix_pass);

            let local_offset = atomicAdd(&s_local_histo[digit], 1u);

            let global_hist = atomicLoad(&histograms[global_offset + digit * num_wgs + wg]);
            let dst = global_hist + local_offset;

            if bool(sort_infos.even_pass) {
                assist_a[dst] = key;
                assist_b[dst] = value;
            } else {
                sort_depths[dst] = key;
                sort_indices[dst] = value;
            }
        }
    }
}
