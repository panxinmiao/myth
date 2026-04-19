// Gaussian Splatting — Radix Sort: Prefix Sum Pass
//
// Computes the global exclusive prefix sum over the histogram bins
// produced by the histogram pass.

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

var<workgroup> s_prefix: array<u32, 256>;

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

    let histo_idx = wid.x * WG_SIZE + lid.x;
    let global_offset = radix_pass * num_wgs * 256u;

    var val = 0u;
    if histo_idx < 256u * num_wgs {
        val = atomicLoad(&histograms[global_offset + histo_idx]);
    }

    // Hillis–Steele inclusive prefix sum
    s_prefix[lid.x] = val;
    workgroupBarrier();

    for (var stride = 1u; stride < WG_SIZE; stride *= 2u) {
        var temp = 0u;
        if lid.x >= stride {
            temp = s_prefix[lid.x - stride];
        }
        workgroupBarrier();
        s_prefix[lid.x] += temp;
        workgroupBarrier();
    }

    // Convert to exclusive and write back
    var prefix = 0u;
    if lid.x > 0u {
        prefix = s_prefix[lid.x - 1u];
    }

    if histo_idx < 256u * num_wgs {
        atomicStore(&histograms[global_offset + histo_idx], prefix);
    }
}
