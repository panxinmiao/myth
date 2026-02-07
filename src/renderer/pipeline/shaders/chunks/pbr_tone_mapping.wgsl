let start_compression = 0.8 - 0.04;
let desaturation = 0.15;

let x = min(out_color.r, min(out_color.g, out_color.b));
let offset = select(0.04, x - 6.25 * x * x, x < 0.08);

var mapped_color = out_color - vec3<f32>(offset);

let peak = max(mapped_color.r, max(mapped_color.g, mapped_color.b));

if (peak < start_compression) {
    out_color =  mapped_color;
}else{
    let d = 1.0 - start_compression;
    let new_peak = 1.0 - d * d / (peak + d - start_compression);

    mapped_color *= (new_peak / peak);

    let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);

    out_color =  mix(mapped_color, vec3<f32>(new_peak), g);
}