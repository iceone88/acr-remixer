// SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
// SPDX-License-Identifier: MIT

//! Embed the frozen backend (packaging/dist/stagegen-backend.exe) when it has
//! been built; without it the GUI runs the Python backend (development).
//! Also records the git commit for the about/bug-report text.
fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let p = root.join("../packaging/dist/stagegen-backend.exe");
    println!("cargo::rustc-check-cfg=cfg(embedded_backend)");
    println!("cargo:rerun-if-changed={}", p.display());
    if p.is_file() {
        println!("cargo:rustc-cfg=embedded_backend");
    }
    let hash = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=GIT_HASH={hash}");
    println!("cargo:rerun-if-changed=../.git/HEAD");
}
