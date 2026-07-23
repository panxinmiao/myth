// Portable Gaussian Splatting Radix Sort
//
// This sorter follows the wait-free hierarchical scan described by
// WebSplatter (arXiv:2602.03207), combined with the 4-bit radix and
// multi-element bitmask ranking used by PlayCanvas' portable sorter.
// Every synchronization point is workgroup-local; no workgroup waits for
// another workgroup to make progress.
//
// Portions of the algorithm are adapted from PlayCanvas Engine, licensed
// under the MIT License.

const sort_wg_size_x: u32 = {{ SORT_WG_SIZE_X }}u;
const sort_wg_size_y: u32 = {{ SORT_WG_SIZE_Y }}u;
const sort_threads_per_wg: u32 = {{ SORT_THREADS_PER_WG }}u;
const sort_items_per_thread: u32 = {{ SORT_ITEMS_PER_THREAD }}u;
const sort_radix_bits: u32 = {{ SORT_RADIX_BITS }}u;
const sort_radix_size: u32 = {{ SORT_RADIX_SIZE }}u;
const sort_scan_items_per_wg: u32 = {{ SORT_SCAN_ITEMS_PER_WG }}u;
const sort_keys_per_wg: u32 = sort_threads_per_wg * sort_items_per_thread;
const sort_mask_words_per_digit: u32 = sort_threads_per_wg / 32u;

// Keep pass-specific values in the generated WGSL source. Safari/WebKit can
// miscompile multiple pipeline variants that differ only by override values.
const radix_shift: u32 = {{ SORT_RADIX_SHIFT }}u;
const write_keys: u32 = {{ SORT_WRITE_KEYS }}u;
const scan_level: u32 = {{ SORT_SCAN_LEVEL }}u;

struct SortInfos {
    keys_size: u32,
    max_workgroups: u32,
    scan_levels: u32,
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
};

@group(0) @binding(0)
var<storage, read> infos: SortInfos;
@group(0) @binding(1)
var<storage, read_write> internal_mem: array<u32>;
@group(0) @binding(2)
var<storage, read> input_keys: array<u32>;
@group(0) @binding(3)
var<storage, read_write> output_keys: array<u32>;
@group(0) @binding(4)
var<storage, read> input_payloads: array<u32>;
@group(0) @binding(5)
var<storage, read_write> output_payloads: array<u32>;

fn ceil_div(value: u32, divisor: u32) -> u32 {
    return (value + divisor - 1u) / divisor;
}

fn scan_level_words(level: u32) -> u32 {
    var words = sort_radix_size * infos.max_workgroups;
    for (var current = 0u; current < level; current += 1u) {
        words = ceil_div(words, sort_scan_items_per_wg);
    }
    return words;
}

fn scan_level_offset(level: u32) -> u32 {
    var offset = 0u;
    var words = sort_radix_size * infos.max_workgroups;
    for (var current = 0u; current < level; current += 1u) {
        offset += words;
        words = ceil_div(words, sort_scan_items_per_wg);
    }
    return offset;
}

var<workgroup> block_histograms: array<atomic<u32>, {{ SORT_RADIX_SIZE }}>;

@compute @workgroup_size({{ SORT_WG_SIZE_X }}, {{ SORT_WG_SIZE_Y }}, 1)
fn block_histogram(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    if local_index < sort_radix_size {
        atomicStore(&block_histograms[local_index], 0u);
    }
    workgroupBarrier();

    let block_base = workgroup_id.x * sort_keys_per_wg;
    for (var item = 0u; item < sort_items_per_thread; item += 1u) {
        let index = block_base + item * sort_threads_per_wg + local_index;
        if index < infos.keys_size {
            let digit = (input_keys[index] >> radix_shift) & (sort_radix_size - 1u);
            atomicAdd(&block_histograms[digit], 1u);
        }
    }

    workgroupBarrier();
    if local_index < sort_radix_size {
        let histogram_index =
            local_index * infos.max_workgroups + workgroup_id.x;
        internal_mem[histogram_index] =
            atomicLoad(&block_histograms[local_index]);
    }
}

var<workgroup> scan_temp: array<u32, {{ SORT_SCAN_ITEMS_PER_WG }}>;

@compute @workgroup_size({{ SORT_WG_SIZE_X }}, {{ SORT_WG_SIZE_Y }}, 1)
fn prefix_scan(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let level_words = scan_level_words(scan_level);
    let level_offset = scan_level_offset(scan_level);
    let block_base = workgroup_id.x * sort_scan_items_per_wg;
    let local_first = local_index * 2u;
    let first = block_base + local_first;
    let second = first + 1u;

    scan_temp[local_first] = 0u;
    scan_temp[local_first + 1u] = 0u;
    if first < level_words {
        scan_temp[local_first] = internal_mem[level_offset + first];
    }
    if second < level_words {
        scan_temp[local_first + 1u] = internal_mem[level_offset + second];
    }

    var offset = 1u;
    for (
        var scan_width = sort_scan_items_per_wg >> 1u;
        scan_width > 0u;
        scan_width >>= 1u
    ) {
        workgroupBarrier();
        if local_index < scan_width {
            let left = offset * (2u * local_index + 1u) - 1u;
            let right = offset * (2u * local_index + 2u) - 1u;
            scan_temp[right] += scan_temp[left];
        }
        offset <<= 1u;
    }

    workgroupBarrier();
    if local_index == 0u {
        let total = scan_temp[sort_scan_items_per_wg - 1u];
        scan_temp[sort_scan_items_per_wg - 1u] = 0u;
        internal_mem[level_offset + level_words + workgroup_id.x] = total;
    }

    for (
        var scan_width = 1u;
        scan_width < sort_scan_items_per_wg;
        scan_width <<= 1u
    ) {
        offset >>= 1u;
        workgroupBarrier();
        if local_index < scan_width {
            let left = offset * (2u * local_index + 1u) - 1u;
            let right = offset * (2u * local_index + 2u) - 1u;
            let left_value = scan_temp[left];
            scan_temp[left] = scan_temp[right];
            scan_temp[right] += left_value;
        }
    }

    workgroupBarrier();
    if first < level_words {
        internal_mem[level_offset + first] = scan_temp[local_first];
    }
    if second < level_words {
        internal_mem[level_offset + second] = scan_temp[local_first + 1u];
    }
}

@compute @workgroup_size({{ SORT_WG_SIZE_X }}, {{ SORT_WG_SIZE_Y }}, 1)
fn prefix_add(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let level_words = scan_level_words(scan_level);
    let level_offset = scan_level_offset(scan_level);
    let parent_offset = level_offset + level_words;
    let block_base = workgroup_id.x * sort_scan_items_per_wg;
    let first = block_base + local_index * 2u;
    let second = first + 1u;
    let addend = internal_mem[parent_offset + workgroup_id.x];

    if first < level_words {
        internal_mem[level_offset + first] += addend;
    }
    if second < level_words {
        internal_mem[level_offset + second] += addend;
    }
}

var<workgroup> digit_masks:
    array<atomic<u32>, {{ SORT_RADIX_SIZE }} * ({{ SORT_THREADS_PER_WG }} / 32)>;
var<workgroup> digit_offsets: array<u32, {{ SORT_RADIX_SIZE }}>;

fn rank_in_round(digit: u32, local_index: u32) -> u32 {
    let word_index = local_index >> 5u;
    let bit_index = local_index & 31u;
    let mask_base = digit * sort_mask_words_per_digit;
    var rank = digit_offsets[digit];

    for (var word = 0u; word < word_index; word += 1u) {
        rank += countOneBits(atomicLoad(&digit_masks[mask_base + word]));
    }

    let current_word = atomicLoad(&digit_masks[mask_base + word_index]);
    let lower_bits = (1u << bit_index) - 1u;
    rank += countOneBits(current_word & lower_bits);
    return rank;
}

@compute @workgroup_size({{ SORT_WG_SIZE_X }}, {{ SORT_WG_SIZE_Y }}, 1)
fn ranked_scatter(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    if local_index < sort_radix_size {
        digit_offsets[local_index] = 0u;
    }
    for (
        var mask_index = local_index;
        mask_index < sort_radix_size * sort_mask_words_per_digit;
        mask_index += sort_threads_per_wg
    ) {
        atomicStore(&digit_masks[mask_index], 0u);
    }
    workgroupBarrier();

    let block_base = workgroup_id.x * sort_keys_per_wg;
    for (var item = 0u; item < sort_items_per_thread; item += 1u) {
        let index = block_base + item * sort_threads_per_wg + local_index;
        let valid = index < infos.keys_size;
        var key = 0u;
        var payload = 0u;
        var digit = 0u;

        if valid {
            key = input_keys[index];
            payload = input_payloads[index];
            digit = (key >> radix_shift) & (sort_radix_size - 1u);
            let mask_index =
                digit * sort_mask_words_per_digit + (local_index >> 5u);
            atomicOr(&digit_masks[mask_index], 1u << (local_index & 31u));
        }

        workgroupBarrier();
        if valid {
            let local_rank = rank_in_round(digit, local_index);
            let histogram_index =
                digit * infos.max_workgroups + workgroup_id.x;
            let output_index = internal_mem[histogram_index] + local_rank;
            if write_keys != 0u {
                output_keys[output_index] = key;
            }
            output_payloads[output_index] = payload;
        }

        if item + 1u < sort_items_per_thread {
            workgroupBarrier();
            if local_index < sort_radix_size {
                let mask_base = local_index * sort_mask_words_per_digit;
                var count = 0u;
                for (
                    var word = 0u;
                    word < sort_mask_words_per_digit;
                    word += 1u
                ) {
                    let mask_index = mask_base + word;
                    count += countOneBits(atomicLoad(&digit_masks[mask_index]));
                    atomicStore(&digit_masks[mask_index], 0u);
                }
                digit_offsets[local_index] += count;
            }
            workgroupBarrier();
        }
    }
}
