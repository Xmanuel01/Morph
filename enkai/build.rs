use std::env;

fn main() {
    let cli_version = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".to_string());
    let language_version = env::var("ENKAI_LANG_VERSION_OVERRIDE").unwrap_or(cli_version);
    println!("cargo:rustc-env=ENKAI_LANG_VERSION={}", language_version);
}
