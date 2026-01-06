// SPDX-License-Identifier: Apache-2.0

// build.rs for hwx
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn find_nvcc() -> Option<String> {
    let nvcc_candidates = [
        "nvcc",
        "/usr/local/cuda/bin/nvcc",
        "/opt/cuda/bin/nvcc",
        "/usr/bin/nvcc",
    ];

    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let cuda_nvcc = format!("{}/bin/nvcc", cuda_home);
        if Command::new(&cuda_nvcc).arg("--version").output().is_ok() {
            return Some(cuda_nvcc);
        }
    }

    for nvcc in &nvcc_candidates {
        if Command::new(nvcc).arg("--version").output().is_ok() {
            return Some(nvcc.to_string());
        }
    }

    None
}

fn detect_gpu_arch() -> String {
    if let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        .output()
    {
        let cap_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !cap_str.is_empty() && cap_str != "N/A" {
            if let Some((major, minor)) = cap_str.split_once('.') {
                return format!("sm_{}{}", major, minor);
            }
        }
    }
    "sm_70".to_string()
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_cuda)");
    let nvcc_opt = find_nvcc();

    if nvcc_opt.is_some() {
        println!("cargo:rustc-cfg=has_cuda");
        let cuda_home = env::var("CUDA_HOME")
            .or_else(|_| env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());

        let driver_candidates = [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/local/nvidia/lib64",
            "/usr/lib/wsl/lib",
        ];

        for dir in &driver_candidates {
            let p = Path::new(dir);
            if p.join("libcuda.so.1").exists() {
                println!("cargo:rustc-link-search=native={}", p.display());
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", p.display());
            }
        }

        let cuda_lib64 = Path::new(&cuda_home).join("lib64");
        if cuda_lib64.exists() {
            println!("cargo:rustc-link-search=native={}", cuda_lib64.display());
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_lib64.display());
        }

        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else {
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-arg=-Wl,-undefined,dynamic_lookup");
        } else {
            println!("cargo:rustc-link-arg=-Wl,--allow-undefined");
        }
    }

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let out_dir = env::var("OUT_DIR").unwrap();
    let cuda_source = format!("{}/src/cub_wrapper.cu", manifest_dir);
    let cuda_output = format!("{}/libcub_sort.so", out_dir);
    let hash_file = format!("{}/cub_wrapper.hash", out_dir);

    if let Ok(bytes) = fs::read(&cuda_source) {
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let prev_hash = fs::read_to_string(&hash_file).unwrap_or_default();
        let rebuild_needed = hash != prev_hash || !Path::new(&cuda_output).exists();

        if rebuild_needed {
            if let Some(nvcc) = &nvcc_opt {
                let gpu_arch = detect_gpu_arch();
                let output = Command::new(nvcc)
                    .args([
                        "-shared",
                        "-o",
                        &cuda_output,
                        &cuda_source,
                        "-lcudart",
                        "-O3",
                        &format!("-arch={}", gpu_arch),
                        "-Xcompiler",
                        "-fPIC",
                    ])
                    .output()
                    .expect("Failed to invoke NVCC");

                if output.status.success() {
                    fs::write(&hash_file, hash).ok();
                }
            }
        }
    }

    if nvcc_opt.is_some() {
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=dylib=cub_sort");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out_dir);
    }
}
