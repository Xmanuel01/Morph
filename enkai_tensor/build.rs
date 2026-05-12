use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_tool(tool: &str, home_vars: &[&str], subdir: &str) -> Option<PathBuf> {
    for home_var in home_vars {
        if let Ok(home) = env::var(home_var) {
            let candidate = Path::new(&home).join(subdir).join(tool);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    let path = env::var_os("PATH")?;
    for entry in env::split_paths(&path) {
        let candidate = entry.join(tool);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn find_nvcc() -> Option<PathBuf> {
    find_tool(
        if cfg!(windows) { "nvcc.exe" } else { "nvcc" },
        &["CUDA_HOME", "CUDA_PATH"],
        "bin",
    )
}

fn find_hipcc() -> Option<PathBuf> {
    find_tool(
        if cfg!(windows) { "hipcc.exe" } else { "hipcc" },
        &["ROCM_HOME", "HIP_PATH", "ROCM_PATH"],
        "bin",
    )
}

fn find_xcrun() -> Option<PathBuf> {
    find_tool("xcrun", &[], "")
}

fn build_cuda() {
    let Some(nvcc) = find_nvcc() else {
        panic!("cuda-kernels feature requires nvcc on PATH or CUDA_HOME/CUDA_PATH; install the CUDA toolkit before claiming first-party CUDA kernel support");
    };
    println!(
        "cargo:warning=building Enkai first-party CUDA kernels with {}",
        nvcc.display()
    );

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .file("cuda/enkai_kernels.cu")
        .flag("-std=c++17")
        .flag("--expt-relaxed-constexpr");
    if cfg!(windows) {
        build.flag("-Xcompiler=/EHsc");
    } else {
        build.flag("-Xcompiler=-fPIC");
    }
    build.compile("enkai_cuda_kernels");

    if let Ok(home) = env::var("CUDA_HOME").or_else(|_| env::var("CUDA_PATH")) {
        let home = Path::new(&home);
        if cfg!(windows) {
            println!(
                "cargo:rustc-link-search=native={}",
                home.join("lib").join("x64").display()
            );
        } else {
            println!(
                "cargo:rustc-link-search=native={}",
                home.join("lib64").display()
            );
            println!(
                "cargo:rustc-link-search=native={}",
                home.join("lib").display()
            );
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
}

fn build_rocm() {
    let Some(hipcc) = find_hipcc() else {
        panic!("rocm-kernels feature requires hipcc on PATH or ROCM_HOME/HIP_PATH/ROCM_PATH; install ROCm before claiming first-party ROCm kernel support");
    };
    println!(
        "cargo:warning=building Enkai first-party ROCm kernels with {}",
        hipcc.display()
    );
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .compiler(hipcc)
        .file("rocm/enkai_kernels.hip.cpp")
        .flag("-std=c++17");
    if cfg!(windows) {
        build.flag("-fPIC");
    } else {
        build.flag("-fPIC");
    }
    build.compile("enkai_rocm_kernels");

    if let Ok(home) = env::var("ROCM_HOME").or_else(|_| env::var("ROCM_PATH")) {
        println!(
            "cargo:rustc-link-search=native={}",
            Path::new(&home).join("lib").display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            Path::new(&home).join("lib64").display()
        );
    }
    if let Ok(hip) = env::var("HIP_PATH") {
        println!(
            "cargo:rustc-link-search=native={}",
            Path::new(&hip).join("lib").display()
        );
    }
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}

fn build_metal() {
    if !cfg!(target_os = "macos") {
        panic!("metal-kernels feature requires macOS with the Xcode Metal toolchain");
    }
    let Some(xcrun) = find_xcrun() else {
        panic!("metal-kernels feature requires xcrun/metal; install Xcode command line tools");
    };
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR missing"));
    let air = out_dir.join("enkai_kernels.air");
    let metallib = out_dir.join("enkai_kernels.metallib");
    let metal_status = Command::new(&xcrun)
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-c",
            "metal/enkai_kernels.metal",
            "-o",
        ])
        .arg(&air)
        .status()
        .expect("failed to run xcrun metal");
    if !metal_status.success() {
        panic!("xcrun metal failed while compiling Enkai Metal kernels");
    }
    let lib_status = Command::new(&xcrun)
        .args(["-sdk", "macosx", "metallib"])
        .arg(&air)
        .arg("-o")
        .arg(&metallib)
        .status()
        .expect("failed to run xcrun metallib");
    if !lib_status.success() {
        panic!("xcrun metallib failed while linking Enkai Metal kernels");
    }
    println!("cargo:rustc-env=ENKAI_METAL_LIBRARY={}", metallib.display());
    println!("cargo:rustc-link-lib=framework=Metal");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/enkai_kernels.cu");
    println!("cargo:rerun-if-changed=rocm/enkai_kernels.hip.cpp");
    println!("cargo:rerun-if-changed=metal/enkai_kernels.metal");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_HOME");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");

    if env::var_os("CARGO_FEATURE_CUDA_KERNELS").is_some() {
        build_cuda();
    }
    if env::var_os("CARGO_FEATURE_ROCM_KERNELS").is_some() {
        build_rocm();
    }
    if env::var_os("CARGO_FEATURE_METAL_KERNELS").is_some() {
        build_metal();
    }
}
