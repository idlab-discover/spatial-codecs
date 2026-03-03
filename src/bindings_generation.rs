use interoptopus::Interop;
use interoptopus::util::NamespaceMappings;
use interoptopus::writer::IndentWriter;
use interoptopus_backend_csharp::{Config as ConfigCS, Generator as GeneratorCS, overloads::DotNet};
use interoptopus_backend_c::{Config as ConfigC, Generator as GeneratorC};

use std::fs;
use std::path::Path;

pub fn generate_csharp_bindings<P: AsRef<Path>>(out_file: P) -> Result<bool, Box<dyn std::error::Error>> {
    let mut generator = GeneratorCS::new(
        ConfigCS {
            class: "SpatialCodecsInterop".to_string(),
            dll_name: "spatial_codecs".to_string(),
            namespace_mappings: NamespaceMappings::new("Be.Ugent"),
            ..ConfigCS::default()
        },
        crate::ffi::build_binding_inventory(),
    );

    generator.add_overload_writer(DotNet::new());
    //generator.add_overload_writer(Unity::new());
    let mut buffer = Vec::new();
    {
        let mut writer = IndentWriter::new(&mut buffer);
        generator.write_to(&mut writer)?;
    }

    let new_contents = String::from_utf8(buffer)?;
    let out_path = out_file.as_ref();

    let unchanged = fs::read_to_string(out_path)
        .map(|existing| existing == new_contents)
        .unwrap_or(false);

    if unchanged {
        return Ok(false);
    }

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(out_path, new_contents)?;
    Ok(true)
}

pub fn generate_c_bindings<P: AsRef<Path>>(out_file: P) -> Result<bool, Box<dyn std::error::Error>> {
    let generator = GeneratorC::new(
        ConfigC {
            ifndef: "SPATIAL_CODECS_INTEROPS_H".to_string(),
            ..ConfigC::default()
        },
        crate::ffi::build_binding_inventory(),
    );

    let mut buffer = Vec::new();
    {
        let mut writer = IndentWriter::new(&mut buffer);
        generator.write_to(&mut writer)?;
    }

    let new_contents = String::from_utf8(buffer)?;
    let out_path = out_file.as_ref();

    let unchanged = fs::read_to_string(out_path)
        .map(|existing| existing == new_contents)
        .unwrap_or(false);

    if unchanged {
        return Ok(false);
    }

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(out_path, new_contents)?;
    Ok(true)
}