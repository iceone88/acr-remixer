//! Embed the frozen backend (packaging/dist/stagegen-backend.exe) when it has
//! been built; without it the GUI runs the Python backend (development).
fn main() {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../packaging/dist/stagegen-backend.exe");
    println!("cargo::rustc-check-cfg=cfg(embedded_backend)");
    println!("cargo:rerun-if-changed={}", p.display());
    if p.is_file() {
        println!("cargo:rustc-cfg=embedded_backend");
    }
}
