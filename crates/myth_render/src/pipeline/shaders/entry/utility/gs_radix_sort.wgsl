// Gaussian Splatting Radix Sort Shader
//
// Ported from web-splat's GPU radix sorter and adapted to Myth's shader
// template system. Sorting is performed on 32-bit depth keys, while payload_a
// contains the splat indices consumed by the render pass.

const histogram_sg_size: u32 = {{ HISTOGRAM_SG_SIZE }}u;
const histogram_wg_size: u32 = {{ HISTOGRAM_WG_SIZE }}u;
const rs_radix_log2: u32 = {{ RS_RADIX_LOG2 }}u;
const rs_radix_size: u32 = {{ RS_RADIX_SIZE }}u;
const rs_keyval_size: u32 = {{ RS_KEYVAL_SIZE }}u;
const rs_histogram_block_rows: u32 = {{ RS_HISTOGRAM_BLOCK_ROWS }}u;
const rs_scatter_block_rows: u32 = {{ RS_SCATTER_BLOCK_ROWS }}u;
const rs_mem_dwords: u32 = {{ RS_MEM_DWORDS }}u;
const rs_mem_sweep_0_offset: u32 = {{ RS_MEM_SWEEP_0_OFFSET }}u;
const rs_mem_sweep_1_offset: u32 = {{ RS_MEM_SWEEP_1_OFFSET }}u;
const rs_mem_sweep_2_offset: u32 = {{ RS_MEM_SWEEP_2_OFFSET }}u;

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
@group(0) @binding(1)
var<storage, read_write> internal_mem: array<atomic<u32>>;
@group(0) @binding(2)
var<storage, read_write> keys_a: array<u32>;
@group(0) @binding(3)
var<storage, read_write> keys_b: array<u32>;
@group(0) @binding(4)
var<storage, read_write> payload_a: array<u32>;
@group(0) @binding(5)
var<storage, read_write> payload_b: array<u32>;

@compute @workgroup_size({{ HISTOGRAM_WG_SIZE }})
fn zero_histograms(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    // if gid.x == 0u {
    //     infos.even_pass = 0u;
    //     infos.odd_pass = 1u;
    // }

    let scatter_block_kvs = histogram_wg_size * rs_scatter_block_rows;
    let scatter_blocks_ru = (infos.keys_size + scatter_block_kvs - 1u) / scatter_block_kvs;

    let histo_size = rs_radix_size;
    var n = (rs_keyval_size + scatter_blocks_ru - 1u) * histo_size;
    let histogram_words = n;
    if infos.keys_size < infos.padded_size {
        n += infos.padded_size - infos.keys_size;
    }

    let line_size = nwg.x * histogram_wg_size;
    for (var cur_index = gid.x; cur_index < n; cur_index += line_size) {
        if cur_index < histogram_words {
            atomicStore(&internal_mem[cur_index], 0u);
        } else {
            keys_a[infos.keys_size + cur_index - histogram_words] = 0xFFFFFFFFu;
        }
    }
}

var<workgroup> smem: array<atomic<u32>, {{ RS_RADIX_SIZE }}>;
var<private> kv: array<u32, {{ RS_HISTOGRAM_BLOCK_ROWS }}>;

fn zero_smem(lid: u32) {
    if lid < rs_radix_size {
        atomicStore(&smem[lid], 0u);
    }
}

fn histogram_pass(pass_: u32, lid: u32) {
    zero_smem(lid);
    workgroupBarrier();

    for (var j = 0u; j < rs_histogram_block_rows; j++) {
        let u_val = bitcast<u32>(kv[j]);
        let digit = extractBits(u_val, pass_ * rs_radix_log2, rs_radix_log2);
        atomicAdd(&smem[digit], 1u);
    }

    workgroupBarrier();

    let histogram_offset = rs_radix_size * pass_ + lid;
    if lid < rs_radix_size {
        atomicAdd(&internal_mem[histogram_offset], atomicLoad(&smem[lid]));
    }
}

fn fill_kv_keys_a(wid: u32, lid: u32) {
    let rs_block_keyvals = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + lid;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_wg_size;
        kv[i] = keys_a[pos];
    }
}

@compute @workgroup_size({{ HISTOGRAM_WG_SIZE }})
fn calculate_histogram(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    fill_kv_keys_a(wid.x, lid.x);

    histogram_pass(3u, lid.x);
    histogram_pass(2u, lid.x);
    histogram_pass(1u, lid.x);
    histogram_pass(0u, lid.x);
}

fn prefix_reduce_smem(lid: u32) {
    var offset = 1u;
    for (var d = rs_radix_size >> 1u; d > 0u; d = d >> 1u) {
        workgroupBarrier();
        if lid < d {
            let ai = offset * (2u * lid + 1u) - 1u;
            let bi = offset * (2u * lid + 2u) - 1u;
            atomicAdd(&smem[bi], atomicLoad(&smem[ai]));
        }
        offset = offset << 1u;
    }

    if lid == 0u {
        atomicStore(&smem[rs_radix_size - 1u], 0u);
    }

    for (var d = 1u; d < rs_radix_size; d = d << 1u) {
        offset = offset >> 1u;
        workgroupBarrier();
        if lid < d {
            let ai = offset * (2u * lid + 1u) - 1u;
            let bi = offset * (2u * lid + 2u) - 1u;
            let t = atomicLoad(&smem[ai]);
            atomicStore(&smem[ai], atomicLoad(&smem[bi]));
            atomicAdd(&smem[bi], t);
        }
    }
}

@compute @workgroup_size({{ PREFIX_WG_SIZE }})
fn prefix_histogram(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let histogram_base = (rs_keyval_size - 1u - wid.x) * rs_radix_size;
    let histogram_offset = histogram_base + lid.x;

    atomicStore(&smem[lid.x], atomicLoad(&internal_mem[histogram_offset]));
    atomicStore(&smem[lid.x + {{ PREFIX_WG_SIZE }}u], atomicLoad(&internal_mem[histogram_offset + {{ PREFIX_WG_SIZE }}u]));

    prefix_reduce_smem(lid.x);
    workgroupBarrier();

    atomicStore(&internal_mem[histogram_offset], atomicLoad(&smem[lid.x]));
    atomicStore(&internal_mem[histogram_offset + {{ PREFIX_WG_SIZE }}u], atomicLoad(&smem[lid.x + {{ PREFIX_WG_SIZE }}u]));
}

var<workgroup> scatter_smem: array<u32, {{ RS_MEM_DWORDS }}>;

fn partitions_base_offset() -> u32 {
    return rs_keyval_size * rs_radix_size;
}

fn histogram_load(digit: u32) -> u32 {
    return atomicLoad(&smem[digit]);
}

fn histogram_store(digit: u32, count: u32) {
    atomicStore(&smem[digit], count);
}

const rs_partition_mask_status: u32 = 0xC0000000u;
const rs_partition_mask_count: u32 = 0x3FFFFFFFu;

var<private> kr: array<u32, {{ RS_SCATTER_BLOCK_ROWS }}>;
var<private> pv: array<u32, {{ RS_SCATTER_BLOCK_ROWS }}>;

fn fill_even_kv(wid: u32, lid: u32) {
    let subgroup_id = lid / histogram_sg_size;
    let subgroup_invoc_id = lid - subgroup_id * histogram_sg_size;
    let subgroup_keyvals = rs_scatter_block_rows * histogram_sg_size;
    let rs_block_keyvals = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + subgroup_id * subgroup_keyvals + subgroup_invoc_id;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        kv[i] = keys_a[pos];
        pv[i] = payload_a[pos];
    }
}

fn fill_odd_kv(wid: u32, lid: u32) {
    let subgroup_id = lid / histogram_sg_size;
    let subgroup_invoc_id = lid - subgroup_id * histogram_sg_size;
    let subgroup_keyvals = rs_scatter_block_rows * histogram_sg_size;
    let rs_block_keyvals = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + subgroup_id * subgroup_keyvals + subgroup_invoc_id;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        kv[i] = keys_b[pos];
        pv[i] = payload_b[pos];
    }
}

fn scatter(
    pass_: u32,
    lid: vec3<u32>,
    gid: vec3<u32>,
    wid: vec3<u32>,
    nwg: vec3<u32>,
    partition_status_invalid: u32,
    partition_status_reduction: u32,
    partition_status_prefix: u32,
) {
    let partition_mask_invalid = partition_status_invalid << 30u;
    let partition_mask_reduction = partition_status_reduction << 30u;
    let partition_mask_prefix = partition_status_prefix << 30u;

    let subgroup_id = lid.x / histogram_sg_size;
    let subgroup_offset = subgroup_id * histogram_sg_size;
    let subgroup_tid = lid.x - subgroup_offset;
    let subgroup_count = {{ SCATTER_WG_SIZE }}u / histogram_sg_size;
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let u_val = bitcast<u32>(kv[i]);
        let digit = extractBits(u_val, pass_ * rs_radix_log2, rs_radix_log2);
        atomicStore(&smem[lid.x], digit);
        var count = 0u;
        var rank = 0u;

        for (var j = 0u; j < histogram_sg_size; j++) {
            if atomicLoad(&smem[subgroup_offset + j]) == digit {
                count += 1u;
                if j <= subgroup_tid {
                    rank += 1u;
                }
            }
        }

        kr[i] = (count << 16u) | rank;
    }

    zero_smem(lid.x);
    workgroupBarrier();

    for (var i = 0u; i < subgroup_count; i++) {
        if subgroup_id == i {
            for (var j = 0u; j < rs_scatter_block_rows; j++) {
                let v = bitcast<u32>(kv[j]);
                let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
                let prev = histogram_load(digit);
                let rank = kr[j] & 0xFFFFu;
                let count = kr[j] >> 16u;
                kr[j] = prev + rank;

                if rank == count {
                    histogram_store(digit, prev + count);
                }
            }
        }
        workgroupBarrier();
    }

    let partition_offset = lid.x + partitions_base_offset();
    let partition_base = wid.x * rs_radix_size;
    if wid.x == 0u {
        let hist_offset = pass_ * rs_radix_size + lid.x;
        if lid.x < rs_radix_size {
            let exc = atomicLoad(&internal_mem[hist_offset]);
            let red = histogram_load(lid.x);

            scatter_smem[lid.x] = exc;
            atomicStore(&internal_mem[partition_offset], (exc + red) | partition_mask_prefix);
        }
    } else {
        if lid.x < rs_radix_size && wid.x < nwg.x - 1u {
            let red = histogram_load(lid.x);
            atomicStore(&internal_mem[partition_offset + partition_base], red | partition_mask_reduction);
        }

        if lid.x < rs_radix_size {
            var partition_base_prev = partition_base - rs_radix_size;
            var exc = 0u;

            loop {
                let prev = atomicLoad(&internal_mem[partition_base_prev + partition_offset]);
                if (prev & rs_partition_mask_status) == partition_mask_invalid {
                    continue;
                }

                exc += prev & rs_partition_mask_count;
                if (prev & rs_partition_mask_status) != partition_mask_prefix {
                    partition_base_prev -= rs_radix_size;
                    continue;
                }

                scatter_smem[lid.x] = exc;
                if wid.x < nwg.x - 1u {
                    atomicAdd(&internal_mem[partition_offset + partition_base], exc | (1u << 30u));
                }
                break;
            }
        }
    }

    prefix_reduce_smem(lid.x);
    workgroupBarrier();

    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let v = bitcast<u32>(kv[i]);
        let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
        let exc = histogram_load(digit);
        let idx = exc + kr[i];
        kr[i] |= idx << 16u;
    }
    workgroupBarrier();

    let smem_reorder_offset = rs_radix_size;
    let smem_base = smem_reorder_offset + lid.x;

    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        let smem_idx = smem_reorder_offset + (kr[j] >> 16u) - 1u;
        scatter_smem[smem_idx] = bitcast<u32>(kv[j]);
    }
    workgroupBarrier();

    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        kv[j] = scatter_smem[smem_base + j * {{ SCATTER_WG_SIZE }}u];
    }
    workgroupBarrier();

    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        let smem_idx = smem_reorder_offset + (kr[j] >> 16u) - 1u;
        scatter_smem[smem_idx] = pv[j];
    }
    workgroupBarrier();

    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        pv[j] = scatter_smem[smem_base + j * {{ SCATTER_WG_SIZE }}u];
    }
    workgroupBarrier();

    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let smem_idx = smem_reorder_offset + (kr[i] >> 16u) - 1u;
        scatter_smem[smem_idx] = kr[i];
    }
    workgroupBarrier();

    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        kr[i] = scatter_smem[smem_base + i * {{ SCATTER_WG_SIZE }}u] & 0xFFFFu;
    }

    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let v = bitcast<u32>(kv[i]);
        let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
        let exc = scatter_smem[digit];
        kr[i] += exc - 1u;
    }
}

// @compute @workgroup_size({{ SCATTER_WG_SIZE }})
// fn scatter_even(
//     @builtin(workgroup_id) wid: vec3<u32>,
//     @builtin(local_invocation_id) lid: vec3<u32>,
//     @builtin(global_invocation_id) gid: vec3<u32>,
//     @builtin(num_workgroups) nwg: vec3<u32>,
// ) {
//     if gid.x == 0u {
//         infos.odd_pass = (infos.odd_pass + 1u) % 2u;
//     }
//     let cur_pass = infos.even_pass * 2u;

//     fill_even_kv(wid.x, lid.x);
//     scatter(cur_pass, lid, gid, wid, nwg, 0u, 1u, 2u);

//     for (var i = 0u; i < rs_scatter_block_rows; i++) {
//         keys_b[kr[i]] = kv[i];
//         payload_b[kr[i]] = pv[i];
//     }
// }

// @compute @workgroup_size({{ SCATTER_WG_SIZE }})
// fn scatter_odd(
//     @builtin(workgroup_id) wid: vec3<u32>,
//     @builtin(local_invocation_id) lid: vec3<u32>,
//     @builtin(global_invocation_id) gid: vec3<u32>,
//     @builtin(num_workgroups) nwg: vec3<u32>,
// ) {
//     if gid.x == 0u {
//         infos.even_pass = (infos.even_pass + 1u) % 2u;
//     }
//     let cur_pass = infos.odd_pass * 2u + 1u;

//     fill_odd_kv(wid.x, lid.x);
//     scatter(cur_pass, lid, gid, wid, nwg, 2u, 3u, 0u);

//     for (var i = 0u; i < rs_scatter_block_rows; i++) {
//         keys_a[kr[i]] = kv[i];
//         payload_a[kr[i]] = pv[i];
//     }
// }


@compute @workgroup_size({{ SCATTER_WG_SIZE }})
fn scatter_pass_0(
    @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>
) {
    fill_even_kv(wid.x, lid.x);
    scatter(0u, lid, gid, wid, nwg, 0u, 1u, 2u); // 固化 pass_ = 0u
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        keys_b[kr[i]] = kv[i];
        payload_b[kr[i]] = pv[i];
    }
}

@compute @workgroup_size({{ SCATTER_WG_SIZE }})
fn scatter_pass_1(
    @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>
) {
    fill_odd_kv(wid.x, lid.x);
    scatter(1u, lid, gid, wid, nwg, 2u, 3u, 0u); // 固化 pass_ = 1u
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        keys_a[kr[i]] = kv[i];
        payload_a[kr[i]] = pv[i];
    }
}

@compute @workgroup_size({{ SCATTER_WG_SIZE }})
fn scatter_pass_2(
    @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>
) {
    fill_even_kv(wid.x, lid.x);
    scatter(2u, lid, gid, wid, nwg, 0u, 1u, 2u); // 固化 pass_ = 2u
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        keys_b[kr[i]] = kv[i];
        payload_b[kr[i]] = pv[i];
    }
}

@compute @workgroup_size({{ SCATTER_WG_SIZE }})
fn scatter_pass_3(
    @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>
) {
    fill_odd_kv(wid.x, lid.x);
    scatter(3u, lid, gid, wid, nwg, 2u, 3u, 0u); // 固化 pass_ = 3u
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        keys_a[kr[i]] = kv[i];
        payload_a[kr[i]] = pv[i];
    }
}