use half;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

fn print_usage_and_exit(program_name: &str) -> ! {
    eprintln!("Myth Engine - LUT Backer Tool");
    eprintln!("Usage: {} -i <input.cube> -o <output.bin>", program_name);
    eprintln!(
        "Example: {} -i assets/test.cube -o assets/baked.bin",
        program_name
    );
    process::exit(1);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let program_name = args.first().map(|s| s.as_str()).unwrap_or("lut_baker");

    let mut input_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-i" | "--input" => {
                if i + 1 < args.len() {
                    input_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "-o" | "--output" => {
                if i + 1 < args.len() {
                    output_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "-h" | "--help" => {
                print_usage_and_exit(program_name);
            }
            _ => {}
        }
        i += 1;
    }

    let (input, output) = match (input_path, output_path) {
        (Some(i), Some(o)) => (i, o),
        _ => {
            eprintln!("❌ Error, missing required arguments.\n");
            print_usage_and_exit(program_name);
        }
    };

    println!("🚀 [LUT Baker] Starting bake process...");
    println!("📥 Input:  {}", input.display());
    println!("📤 Output: {}", output.display());

    // 2. Read text file
    let content = match fs::read_to_string(&input) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("❌ Error: Failed to read input file: {}", e);
            process::exit(1);
        }
    };

    let mut float_data: Vec<f32> = Vec::new();

    // 3. Parse lines into f32 values, ignoring comments and empty lines
    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();

        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.chars().next().map_or(false, |c| c.is_alphabetic())
        {
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() == 3 {
            match (
                parts[0].parse::<f32>(),
                parts[1].parse::<f32>(),
                parts[2].parse::<f32>(),
            ) {
                (Ok(r), Ok(g), Ok(b)) => {
                    float_data.push(r);
                    float_data.push(g);
                    float_data.push(b);
                }
                _ => {
                    eprintln!(
                        "⚠️ Warning: Failed to parse float on line {}: {}",
                        line_num + 1,
                        trimmed
                    );
                }
            }
        }
    }

    let rgb_count = float_data.len() / 3;
    if rgb_count == 0 {
        eprintln!("❌ Error: No valid RGB data found in the file.");
        process::exit(1);
    }

    // 4. Convert to Rgba16Float (half-precision) and pack into bytes
    let mut byte_data: Vec<u8> = Vec::with_capacity(rgb_count * 8);

    for chunk in float_data.chunks_exact(3) {
        let r = half::f16::from_f32(chunk[0]);
        let g = half::f16::from_f32(chunk[1]);
        let b = half::f16::from_f32(chunk[2]);
        let a = half::f16::from_f32(1.0); // Alpha set to 1.0

        // Write 8 bytes in little-endian order
        byte_data.extend_from_slice(&r.to_le_bytes());
        byte_data.extend_from_slice(&g.to_le_bytes());
        byte_data.extend_from_slice(&b.to_le_bytes());
        byte_data.extend_from_slice(&a.to_le_bytes());
    }

    println!("✅ Converted to Rgba16Float ({} bytes).", byte_data.len());

    // 5. Write raw binary file
    if let Err(e) = fs::write(&output, &byte_data) {
        eprintln!("❌ Error: Failed to write output file: {}", e);
        process::exit(1);
    }

    println!("🎉 Successfully baked LUT to raw binary!");
}
