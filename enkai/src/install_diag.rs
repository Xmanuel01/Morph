use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
struct InstallDiagnosticsArgs {
    json: bool,
    output: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct InstallDiagnosticsReport {
    schema_version: u32,
    profile: &'static str,
    cli_version: &'static str,
    language_version: &'static str,
    executable_path: String,
    install_root: String,
    bundle_layout_ok: bool,
    bundle_paths: BundlePaths,
    stdlib_file_count: usize,
    examples_file_count: usize,
    native_payloads: NativePayloads,
    bundle_manifest: BundleManifestDiagnostics,
    install_manifest: InstallManifestDiagnostics,
    rust_toolchain_visible: RustToolchainVisible,
    selfhost_entrypoints: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct BundlePaths {
    readme: bool,
    std_dir: bool,
    hello_example: bool,
}

#[derive(Debug, Serialize)]
struct NativePayloads {
    library_name: &'static str,
    install_root_present: bool,
}

#[derive(Debug, Serialize)]
struct InstallManifestDiagnostics {
    path: String,
    present: bool,
    parse_ok: bool,
    version_matches_cli: bool,
    installed_version: Option<String>,
    source_type: Option<String>,
    managed_entries: Vec<String>,
    missing_managed_entries: Vec<String>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct InstallManifest {
    installed_version: String,
    source_type: Option<String>,
    managed_entries: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct BundleManifestDiagnostics {
    path: String,
    present: bool,
    parse_ok: bool,
    version_matches_cli: bool,
    target_os_matches_host: bool,
    entrypoint_matches_host: bool,
    version: Option<String>,
    target_os: Option<String>,
    entrypoint: Option<String>,
    missing_required_paths: Vec<String>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BundleManifest {
    version: String,
    target_os: Option<String>,
    entrypoint: Option<String>,
    required_paths: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct RustToolchainVisible {
    cargo: bool,
    rustc: bool,
}

pub fn print_install_diagnostics_usage() {
    eprintln!("  enkai install-diagnostics [--json] [--output <file>]");
}

pub fn install_diagnostics_command(args: &[String]) -> i32 {
    let parsed = match parse_args(args) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("enkai install-diagnostics: {}", err);
            print_install_diagnostics_usage();
            return 1;
        }
    };

    let report = match build_report() {
        Ok(report) => report,
        Err(err) => {
            eprintln!("enkai install-diagnostics: {}", err);
            return 1;
        }
    };

    let json = match serde_json::to_string_pretty(&report) {
        Ok(value) => value,
        Err(err) => {
            eprintln!(
                "enkai install-diagnostics: failed to serialize diagnostics report: {}",
                err
            );
            return 1;
        }
    };

    if let Some(path) = parsed.output.as_ref() {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(err) = fs::create_dir_all(parent) {
                    eprintln!(
                        "enkai install-diagnostics: failed to create output directory {}: {}",
                        parent.display(),
                        err
                    );
                    return 1;
                }
            }
        }
        if let Err(err) = fs::write(path, json.as_bytes()) {
            eprintln!(
                "enkai install-diagnostics: failed to write report {}: {}",
                path.display(),
                err
            );
            return 1;
        }
    }

    if parsed.json {
        println!("{}", json);
    } else {
        println!(
            "install-root={} bundle_layout_ok={} stdlib_files={} hello_example={}",
            report.install_root,
            report.bundle_layout_ok,
            report.stdlib_file_count,
            report.bundle_paths.hello_example
        );
    }

    if report.bundle_layout_ok {
        0
    } else {
        1
    }
}

fn parse_args(args: &[String]) -> Result<InstallDiagnosticsArgs, String> {
    let mut parsed = InstallDiagnosticsArgs {
        json: false,
        output: None,
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => parsed.json = true,
            "--output" => {
                index += 1;
                if index >= args.len() {
                    return Err("--output requires a value".to_string());
                }
                parsed.output = Some(PathBuf::from(args[index].trim()));
            }
            unknown => return Err(format!("unknown option '{}'", unknown)),
        }
        index += 1;
    }
    Ok(parsed)
}

fn build_report() -> Result<InstallDiagnosticsReport, String> {
    let exe = env::current_exe()
        .map_err(|err| format!("failed to resolve current executable path: {}", err))?;
    let install_root = exe
        .parent()
        .ok_or_else(|| "current executable does not have a parent directory".to_string())?;

    let readme = install_root.join("README.txt");
    let std_dir = install_root.join("std");
    let examples_dir = install_root.join("examples");
    let hello_example = examples_dir.join("hello").join("main.enk");
    let stdlib_file_count = count_files_with_extension(&std_dir, "enk");
    let examples_file_count = count_files_with_extension(&examples_dir, "enk");
    let native_path = install_root.join(native_library_name());
    let bundle_manifest = inspect_bundle_manifest(install_root);
    let install_manifest = inspect_install_manifest(install_root);

    let report = InstallDiagnosticsReport {
        schema_version: 1,
        profile: "install_bundle_diagnostics",
        cli_version: env!("CARGO_PKG_VERSION"),
        language_version: env!("ENKAI_LANG_VERSION"),
        executable_path: exe.display().to_string(),
        install_root: install_root.display().to_string(),
        bundle_layout_ok: readme.is_file() && std_dir.is_dir() && hello_example.is_file(),
        bundle_paths: BundlePaths {
            readme: readme.is_file(),
            std_dir: std_dir.is_dir(),
            hello_example: hello_example.is_file(),
        },
        stdlib_file_count,
        examples_file_count,
        native_payloads: NativePayloads {
            library_name: native_library_name(),
            install_root_present: native_path.is_file(),
        },
        bundle_manifest,
        install_manifest,
        rust_toolchain_visible: RustToolchainVisible {
            cargo: binary_visible("cargo"),
            rustc: binary_visible("rustc"),
        },
        selfhost_entrypoints: vec!["run", "check", "build", "test"],
    };
    Ok(report)
}

fn inspect_bundle_manifest(install_root: &Path) -> BundleManifestDiagnostics {
    let path = install_root.join("bundle_manifest.json");
    if !path.is_file() {
        return BundleManifestDiagnostics {
            path: path.display().to_string(),
            present: false,
            parse_ok: false,
            version_matches_cli: false,
            target_os_matches_host: false,
            entrypoint_matches_host: false,
            version: None,
            target_os: None,
            entrypoint: None,
            missing_required_paths: Vec::new(),
            error: None,
        };
    }
    let raw = match fs::read_to_string(&path) {
        Ok(value) => value.trim_start_matches('\u{feff}').to_string(),
        Err(err) => {
            return BundleManifestDiagnostics {
                path: path.display().to_string(),
                present: true,
                parse_ok: false,
                version_matches_cli: false,
                target_os_matches_host: false,
                entrypoint_matches_host: false,
                version: None,
                target_os: None,
                entrypoint: None,
                missing_required_paths: Vec::new(),
                error: Some(format!("failed to read bundle manifest: {}", err)),
            }
        }
    };
    let manifest: BundleManifest = match serde_json::from_str(&raw) {
        Ok(value) => value,
        Err(err) => {
            return BundleManifestDiagnostics {
                path: path.display().to_string(),
                present: true,
                parse_ok: false,
                version_matches_cli: false,
                target_os_matches_host: false,
                entrypoint_matches_host: false,
                version: None,
                target_os: None,
                entrypoint: None,
                missing_required_paths: Vec::new(),
                error: Some(format!("failed to parse bundle manifest: {}", err)),
            }
        }
    };
    let missing_required_paths = manifest
        .required_paths
        .clone()
        .unwrap_or_default()
        .iter()
        .filter(|entry| !install_root.join(entry).exists())
        .cloned()
        .collect::<Vec<_>>();
    let target_os_matches_host = manifest
        .target_os
        .as_deref()
        .map(|value| value == host_target_os())
        .unwrap_or(false);
    let entrypoint_matches_host = manifest
        .entrypoint
        .as_deref()
        .map(|value| value == host_entrypoint_name())
        .unwrap_or(false);
    BundleManifestDiagnostics {
        path: path.display().to_string(),
        present: true,
        parse_ok: true,
        version_matches_cli: manifest.version == env!("CARGO_PKG_VERSION"),
        target_os_matches_host,
        entrypoint_matches_host,
        version: Some(manifest.version),
        target_os: manifest.target_os,
        entrypoint: manifest.entrypoint,
        missing_required_paths,
        error: None,
    }
}

fn inspect_install_manifest(install_root: &Path) -> InstallManifestDiagnostics {
    let path = install_root.join("install_manifest.json");
    if !path.is_file() {
        return InstallManifestDiagnostics {
            path: path.display().to_string(),
            present: false,
            parse_ok: false,
            version_matches_cli: false,
            installed_version: None,
            source_type: None,
            managed_entries: Vec::new(),
            missing_managed_entries: Vec::new(),
            error: None,
        };
    }

    let raw = match fs::read_to_string(&path) {
        Ok(value) => value.trim_start_matches('\u{feff}').to_string(),
        Err(err) => {
            return InstallManifestDiagnostics {
                path: path.display().to_string(),
                present: true,
                parse_ok: false,
                version_matches_cli: false,
                installed_version: None,
                source_type: None,
                managed_entries: Vec::new(),
                missing_managed_entries: Vec::new(),
                error: Some(format!("failed to read install manifest: {}", err)),
            }
        }
    };

    let manifest: InstallManifest = match serde_json::from_str(&raw) {
        Ok(value) => value,
        Err(err) => {
            return InstallManifestDiagnostics {
                path: path.display().to_string(),
                present: true,
                parse_ok: false,
                version_matches_cli: false,
                installed_version: None,
                source_type: None,
                managed_entries: Vec::new(),
                missing_managed_entries: Vec::new(),
                error: Some(format!("failed to parse install manifest: {}", err)),
            }
        }
    };

    let managed_entries = manifest.managed_entries.unwrap_or_default();
    let missing_managed_entries = managed_entries
        .iter()
        .filter(|entry| !install_root.join(entry).exists())
        .cloned()
        .collect::<Vec<_>>();
    let version_matches_cli = manifest.installed_version == env!("CARGO_PKG_VERSION");
    InstallManifestDiagnostics {
        path: path.display().to_string(),
        present: true,
        parse_ok: true,
        version_matches_cli,
        installed_version: Some(manifest.installed_version),
        source_type: manifest.source_type,
        managed_entries,
        missing_managed_entries,
        error: None,
    }
}

fn binary_visible(name: &str) -> bool {
    let Some(paths) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&paths).any(|entry| executable_exists(&entry, name))
}

fn executable_exists(dir: &Path, name: &str) -> bool {
    if cfg!(windows) {
        let exts = [".exe", ".cmd", ".bat", ""];
        exts.iter()
            .map(|ext| dir.join(format!("{}{}", name, ext)))
            .any(|candidate| candidate.is_file())
    } else {
        dir.join(name).is_file()
    }
}

fn count_files_with_extension(root: &Path, ext: &str) -> usize {
    if !root.is_dir() {
        return 0;
    }
    let mut count = 0usize;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| value.eq_ignore_ascii_case(ext))
                .unwrap_or(false)
            {
                count += 1;
            }
        }
    }
    count
}

fn native_library_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "enkai_native.dll"
    } else if cfg!(target_os = "macos") {
        "libenkai_native.dylib"
    } else {
        "libenkai_native.so"
    }
}

fn host_entrypoint_name() -> &'static str {
    if cfg!(windows) {
        "enkai.exe"
    } else {
        "enkai"
    }
}

fn host_target_os() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux"
    }
}
