// Gaussian Splatting Sort Padding Shader
//
// Fills the inactive tail of the current radix-sort dispatch window with a
// sentinel key and a zero payload so the sorter never observes stale data from
// earlier frames. The row count is specialized at pipeline creation to match
// the device workgroup-storage budget used by the radix-sort passes.

const pad_wg_size: u32 = {{ HISTOGRAM_WG_SIZE }}u;
override RS_HISTOGRAM_BLOCK_ROWS: u32 = 15u;

struct SortInfos {
    keys_size: u32,
    padded_size: u32,
    passes: u32,
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
};

@group(0) @binding(0)
var<storage, read_write> infos: SortInfos;
@group(0) @binding(2)
var<storage, read_write> keys_a: array<u32>;
@group(0) @binding(4)
var<storage, read_write> payload_a: array<u32>;

@compute @workgroup_size({{ HISTOGRAM_WG_SIZE }})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let base_idx = wid.x * pad_wg_size * RS_HISTOGRAM_BLOCK_ROWS + lid.x;

    for (var row = 0u; row < RS_HISTOGRAM_BLOCK_ROWS; row++) {
        let idx = base_idx + row * pad_wg_size;
        if idx >= infos.keys_size && idx < infos.padded_size {
            keys_a[idx] = 0xFFFFFFFFu;
            payload_a[idx] = 0u;
        }
    }
}