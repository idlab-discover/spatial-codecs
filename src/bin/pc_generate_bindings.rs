use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cs_file = PathBuf::from("bindings/csharp/SpatialCodecsInterop.cs");
    let updated = spatial_codecs::bindings_generation::generate_csharp_bindings(&cs_file)?;

    if updated {
        println!("Updated C# bindings at {}", cs_file.display());
    } else {
        println!("C# bindings already up to date at {}", cs_file.display());
    }

    let c_file = PathBuf::from("bindings/c/spatial_codecs_interops.h");
    let updated = spatial_codecs::bindings_generation::generate_c_bindings(&c_file)?;

    if updated {
        println!("Updated C bindings at {}", c_file.display());
    } else {
        println!("C bindings already up to date at {}", c_file.display());
    }

    Ok(())
}
