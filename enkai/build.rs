use std::env;

fn main() {
    let cli_version = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".to_string());
    let language_version = env::var("ENKAI_LANG_VERSION_OVERRIDE").unwrap_or(cli_version);
    println!("cargo:rustc-env=ENKAI_LANG_VERSION={}", language_version);
    println!("cargo:rerun-if-changed=contracts/enkai_chat_v1.proto");
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc");
    env::set_var("PROTOC", protoc);
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["contracts/enkai_chat_v1.proto"], &["contracts"])
        .expect("compile grpc protos");
}
