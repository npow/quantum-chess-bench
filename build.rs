use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn collect_rust_files(directory: &Path, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(directory) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_rust_files(&path, files);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
}

fn digest_files(root: &Path, files: &mut Vec<PathBuf>) -> String {
    files.sort();
    let mut hash = Sha256::new();
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path.strip_prefix(root).unwrap_or(path);
        hash.update(relative.to_string_lossy().as_bytes());
        hash.update([0]);
        hash.update(fs::read(path).expect("hash input must be readable"));
        hash.update([0]);
    }
    format!("{:x}", hash.finalize())
}

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets manifest dir"));
    let mut sources = Vec::new();
    for directory in ["src", "vendor/qchess/src", "vendor/glicko2/src"] {
        collect_rust_files(&root.join(directory), &mut sources);
    }
    println!(
        "cargo:rustc-env=QBENCH_ENGINE_SOURCE_HASH={}",
        digest_files(&root, &mut sources)
    );

    let mut runner = vec![root.join("src/main.rs")];
    println!(
        "cargo:rustc-env=QBENCH_RUNNER_HASH={}",
        digest_files(&root, &mut runner)
    );

    let mut build = [
        "Cargo.toml",
        "Cargo.lock",
        "UPSTREAM.md",
        "build.rs",
        "vendor/qchess/Cargo.toml",
        "vendor/glicko2/Cargo.toml",
    ]
    .into_iter()
    .map(|path| root.join(path))
    .collect::<Vec<_>>();
    println!(
        "cargo:rustc-env=QBENCH_BUILD_HASH={}",
        digest_files(&root, &mut build)
    );
}
