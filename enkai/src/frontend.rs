use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NewKind {
    Backend,
    Service,
    LlmBackend,
    FrontendChat,
    FullstackChat,
    LlmFullstack,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendProfile {
    Backend,
    Service,
    Llm,
}

#[derive(Debug, Clone)]
struct NewOptions {
    kind: NewKind,
    target_dir: PathBuf,
    api_version: String,
    backend_url: String,
    force: bool,
}

#[derive(Debug, Clone)]
struct SdkOptions {
    output: PathBuf,
    api_version: String,
}

pub fn new_command(args: &[String]) -> i32 {
    let options = match parse_new_options(args) {
        Ok(opts) => opts,
        Err(err) => {
            eprintln!("enkai new: {}", err);
            print_new_usage();
            return 1;
        }
    };
    match scaffold_project(&options) {
        Ok(()) => {
            println!("Scaffold created at {}", options.target_dir.display());
            0
        }
        Err(err) => {
            eprintln!("enkai new: {}", err);
            1
        }
    }
}

pub fn sdk_command(args: &[String]) -> i32 {
    let options = match parse_sdk_options(args) {
        Ok(opts) => opts,
        Err(err) => {
            eprintln!("enkai sdk: {}", err);
            print_sdk_usage();
            return 1;
        }
    };
    if let Err(err) = write_sdk_file(
        &options.output,
        &render_typescript_sdk(&options.api_version),
    ) {
        eprintln!("enkai sdk: {}", err);
        return 1;
    }
    println!("SDK generated at {}", options.output.display());
    0
}

pub fn print_new_usage() {
    eprintln!(
        "  enkai new <backend|service|llm-backend|frontend-chat|fullstack-chat|llm-fullstack> <target_dir> [--api-version <v>] [--backend-url <url>] [--force]"
    );
}

pub fn print_sdk_usage() {
    eprintln!("  enkai sdk generate <output_file> [--api-version <v>]");
}

fn parse_new_options(args: &[String]) -> Result<NewOptions, String> {
    if args.len() < 2 {
        return Err("expected scaffold kind and target directory".to_string());
    }
    let kind = parse_new_kind(&args[0])?;
    let target_dir = PathBuf::from(&args[1]);
    let mut api_version = "v1".to_string();
    let mut backend_url = "http://localhost:8080".to_string();
    let mut force = false;
    let mut idx = 2usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--api-version" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--api-version requires a value".to_string());
                }
                api_version = args[idx].trim().to_string();
                if api_version.is_empty() {
                    return Err("--api-version must not be empty".to_string());
                }
            }
            "--backend-url" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--backend-url requires a value".to_string());
                }
                backend_url = args[idx].trim().to_string();
                if backend_url.is_empty() {
                    return Err("--backend-url must not be empty".to_string());
                }
            }
            "--force" => {
                force = true;
            }
            other => {
                return Err(format!("unknown option '{}'", other));
            }
        }
        idx += 1;
    }
    Ok(NewOptions {
        kind,
        target_dir,
        api_version,
        backend_url,
        force,
    })
}

fn parse_new_kind(raw: &str) -> Result<NewKind, String> {
    match raw {
        "backend" => Ok(NewKind::Backend),
        "service" => Ok(NewKind::Service),
        "llm-backend" => Ok(NewKind::LlmBackend),
        "frontend-chat" => Ok(NewKind::FrontendChat),
        "fullstack-chat" => Ok(NewKind::FullstackChat),
        "llm-fullstack" => Ok(NewKind::LlmFullstack),
        _ => Err(format!(
            "unknown scaffold kind '{}'; expected backend|service|llm-backend|frontend-chat|fullstack-chat|llm-fullstack",
            raw
        )),
    }
}

fn parse_sdk_options(args: &[String]) -> Result<SdkOptions, String> {
    if args.is_empty() {
        return Err("expected subcommand (generate)".to_string());
    }
    if args[0] != "generate" {
        return Err(format!(
            "unknown sdk subcommand '{}'; expected generate",
            args[0]
        ));
    }
    if args.len() < 2 {
        return Err("generate requires an output file path".to_string());
    }
    let output = PathBuf::from(&args[1]);
    let mut api_version = "v1".to_string();
    let mut idx = 2usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--api-version" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--api-version requires a value".to_string());
                }
                api_version = args[idx].trim().to_string();
                if api_version.is_empty() {
                    return Err("--api-version must not be empty".to_string());
                }
            }
            other => {
                return Err(format!("unknown option '{}'", other));
            }
        }
        idx += 1;
    }
    Ok(SdkOptions {
        output,
        api_version,
    })
}

fn scaffold_project(options: &NewOptions) -> Result<(), String> {
    prepare_destination(&options.target_dir, options.force)?;
    match options.kind {
        NewKind::Backend => scaffold_backend_project(
            &options.target_dir,
            &options.api_version,
            BackendProfile::Backend,
        ),
        NewKind::Service => scaffold_backend_project(
            &options.target_dir,
            &options.api_version,
            BackendProfile::Service,
        ),
        NewKind::LlmBackend => scaffold_backend_project(
            &options.target_dir,
            &options.api_version,
            BackendProfile::Llm,
        ),
        NewKind::FrontendChat => scaffold_frontend_project(
            &options.target_dir,
            &options.api_version,
            &options.backend_url,
        ),
        NewKind::FullstackChat => scaffold_fullstack_project(options, BackendProfile::Backend),
        NewKind::LlmFullstack => scaffold_fullstack_project(options, BackendProfile::Llm),
    }
}

fn prepare_destination(path: &Path, force: bool) -> Result<(), String> {
    if path.exists() {
        if !path.is_dir() {
            return Err(format!(
                "target path is not a directory: {}",
                path.display()
            ));
        }
        if !force {
            let mut entries = fs::read_dir(path)
                .map_err(|err| format!("failed to inspect {}: {}", path.display(), err))?;
            if entries.next().is_some() {
                return Err(format!(
                    "target directory is not empty: {} (use --force to overwrite files)",
                    path.display()
                ));
            }
        }
        return Ok(());
    }
    fs::create_dir_all(path).map_err(|err| format!("failed to create {}: {}", path.display(), err))
}

fn scaffold_backend_project(
    root: &Path,
    api_version: &str,
    profile: BackendProfile,
) -> Result<(), String> {
    write_text_file(&root.join("enkai.toml"), &render_backend_manifest(root))?;
    write_text_file(
        &root.join("README.md"),
        &render_backend_readme(root, api_version, profile),
    )?;
    write_text_file(
        &root.join(".gitignore"),
        ".env\n.env.local\nlogs/\n*.db\n*.db-journal\n",
    )?;
    write_text_file(
        &root.join(".env.example"),
        &render_backend_env_example(api_version, profile),
    )?;
    write_text_file(
        &root.join("contracts").join("backend_api.snapshot.json"),
        &render_backend_contract_snapshot(api_version),
    )?;
    write_text_file(
        &root
            .join("contracts")
            .join("conversation_state.schema.json"),
        &render_conversation_schema_json(),
    )?;
    write_text_file(
        &root.join("contracts").join("deploy_env.snapshot.json"),
        &render_deploy_env_snapshot(api_version, profile),
    )?;
    write_text_file(
        &root.join("migrations").join("001_conversation_state.sql"),
        MIGRATION_001_CONVERSATION_STATE_SQL,
    )?;
    write_text_file(
        &root
            .join("migrations")
            .join("002_conversation_state_index.sql"),
        MIGRATION_002_CONVERSATION_STATE_INDEX_SQL,
    )?;
    write_text_file(
        &root.join("scripts").join("validate_env_contract.py"),
        &render_backend_env_validator_py(api_version, profile),
    )?;
    write_text_file(
        &root.join("src").join("main.enk"),
        &render_backend_main(api_version),
    )?;
    Ok(())
}

fn scaffold_frontend_project(
    root: &Path,
    api_version: &str,
    backend_url: &str,
) -> Result<(), String> {
    write_text_file(
        &root.join("package.json"),
        &render_frontend_package_json(root),
    )?;
    write_text_file(
        &root.join("README.md"),
        &render_frontend_readme(api_version, backend_url),
    )?;
    write_text_file(
        &root.join(".env.example"),
        &render_frontend_env_example(api_version, backend_url),
    )?;
    write_text_file(
        &root.join(".gitignore"),
        "node_modules/\ndist/\n.env.local\n",
    )?;
    write_text_file(&root.join("index.html"), FRONTEND_INDEX_HTML)?;
    write_text_file(&root.join("tsconfig.json"), FRONTEND_TSCONFIG)?;
    write_text_file(&root.join("tsconfig.node.json"), FRONTEND_TSCONFIG_NODE)?;
    write_text_file(&root.join("vite.config.ts"), FRONTEND_VITE_CONFIG)?;
    write_text_file(&root.join("src").join("main.tsx"), FRONTEND_MAIN_TSX)?;
    write_text_file(&root.join("src").join("styles.css"), FRONTEND_STYLES_CSS)?;
    write_text_file(
        &root.join("src").join("App.tsx"),
        &render_frontend_app_tsx(api_version),
    )?;
    write_text_file(
        &root.join("contracts").join("sdk_api.snapshot.json"),
        &render_sdk_contract_snapshot(api_version),
    )?;
    write_text_file(
        &root.join("src").join("sdk").join("enkaiClient.ts"),
        &render_typescript_sdk(api_version),
    )?;
    write_text_file(&root.join("src").join("types.ts"), FRONTEND_TYPES_TS)?;
    Ok(())
}

fn scaffold_fullstack_project(
    options: &NewOptions,
    backend_profile: BackendProfile,
) -> Result<(), String> {
    let root = &options.target_dir;
    write_text_file(
        &root.join("README.md"),
        &render_fullstack_readme(&options.api_version, &options.backend_url, backend_profile),
    )?;
    write_text_file(
        &root.join(".gitignore"),
        "frontend/node_modules/\nfrontend/dist/\nbackend/logs/\n",
    )?;
    scaffold_backend_project(&root.join("backend"), &options.api_version, backend_profile)?;
    scaffold_frontend_project(
        &root.join("frontend"),
        &options.api_version,
        &options.backend_url,
    )?;
    Ok(())
}

fn write_sdk_file(path: &Path, content: &str) -> Result<(), String> {
    write_text_file(path, content)
}

fn write_text_file(path: &Path, content: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {}", parent.display(), err))?;
    }
    fs::write(path, content).map_err(|err| format!("failed to write {}: {}", path.display(), err))
}

fn render_backend_manifest(root: &Path) -> String {
    let name = sanitize_name(root);
    format!(
        "[package]\nname = \"{}\"\nversion = \"0.1.0\"\n\n[dependencies]\n",
        name
    )
}

fn backend_profile_name(profile: BackendProfile) -> &'static str {
    match profile {
        BackendProfile::Backend => "backend",
        BackendProfile::Service => "service",
        BackendProfile::Llm => "llm-backend",
    }
}

fn backend_profile_slug(profile: BackendProfile) -> &'static str {
    match profile {
        BackendProfile::Backend => "backend",
        BackendProfile::Service => "service",
        BackendProfile::Llm => "llm",
    }
}

fn render_backend_readme(root: &Path, api_version: &str, profile: BackendProfile) -> String {
    let name = sanitize_name(root);
    format!(
        "# {}\n\nEnkai {} scaffold (`v2.2.0` contract freeze).\n\n## API Contract\n\n- Base path: `/api/{}`\n- Header required by SDK: `x-enkai-api-version: {}`\n- Routes:\n  - `GET /api/{}/health`\n  - `POST /api/{}/chat`\n  - `GET /api/{}/chat/stream`\n  - `GET /api/{}/chat/ws`\n\n## Contract Snapshots\n\n- Backend contract snapshot: `contracts/backend_api.snapshot.json`\n- Conversation schema: `contracts/conversation_state.schema.json`\n- Deployment env snapshot: `contracts/deploy_env.snapshot.json`\n\n## Persistence + Migration\n\n- Latest conversation state is persisted to `conversation_state.json`.\n- A backup copy is persisted to `conversation_state.backup.json`.\n- SQLite metadata DB is stored at `ENKAI_CONVERSATION_DB` (default `conversation_state.db`).\n- Startup applies DB migration steps from `migrations/` and records versions in `schema_migrations`.\n\n## Environment Contract\n\n- Copy `.env.example` to `.env` and validate before deploy:\n  - `python scripts/validate_env_contract.py --env-file .env`\n\n## Run\n\n- `enkai serve --host 0.0.0.0 --port 8080 .`\n",
        name,
        backend_profile_name(profile),
        api_version,
        api_version,
        api_version,
        api_version,
        api_version,
        api_version
    )
}

fn render_backend_env_example(api_version: &str, profile: BackendProfile) -> String {
    let mut out = format!(
        "ENKAI_APP_PROFILE={}\nENKAI_API_VERSION={}\nENKAI_SERVE_HOST=0.0.0.0\nENKAI_SERVE_PORT=8080\nENKAI_CONVERSATION_DIR=.\nENKAI_CONVERSATION_DB=conversation_state.db\nENKAI_LOG_PATH=server.jsonl\n",
        backend_profile_slug(profile),
        api_version
    );
    if profile == BackendProfile::Llm {
        out.push_str("ENKAI_MODEL_NAME=tinyllm\nENKAI_MODEL_VERSION=latest\n");
    }
    out
}

fn render_backend_contract_snapshot(api_version: &str) -> String {
    format!(
        "{{\n  \"snapshot_version\": 1,\n  \"api_version\": \"{}\",\n  \"base_path\": \"/api/{}\",\n  \"required_headers\": [\n    \"x-enkai-api-version\"\n  ],\n  \"optional_headers\": [\n    \"authorization\"\n  ],\n  \"routes\": [\n    {{\n      \"method\": \"GET\",\n      \"path\": \"/api/{}/health\"\n    }},\n    {{\n      \"method\": \"POST\",\n      \"path\": \"/api/{}/chat\"\n    }},\n    {{\n      \"method\": \"GET\",\n      \"path\": \"/api/{}/chat/stream\"\n    }},\n    {{\n      \"method\": \"GET\",\n      \"path\": \"/api/{}/chat/ws\"\n    }}\n  ],\n  \"streaming\": {{\n    \"sse\": {{\n      \"content_type\": \"text/event-stream\",\n      \"events\": [\n        \"token\",\n        \"done\"\n      ]\n    }},\n    \"websocket\": {{\n      \"encoding\": \"json-text\",\n      \"events\": [\n        \"token\",\n        \"done\"\n      ]\n    }}\n  }},\n  \"middlewares\": [\n    \"auth\",\n    \"rate_limit\",\n    \"jsonl_log\",\n    \"default\"\n  ],\n  \"error_codes\": [\n    \"missing_api_version_header\",\n    \"api_version_mismatch\",\n    \"missing_prompt\",\n    \"not_found\"\n  ],\n  \"persistence\": {{\n    \"file\": \"conversation_state.json\",\n    \"schema_version\": 1,\n    \"migration_paths\": [\n      \"v0_to_v1\"\n    ]\n  }}\n}}\n",
        api_version, api_version, api_version, api_version, api_version, api_version
    )
}

fn render_sdk_contract_snapshot(api_version: &str) -> String {
    format!(
        "{{\n  \"snapshot_version\": 1,\n  \"api_version\": \"{}\",\n  \"required_headers\": [\n    \"x-enkai-api-version\"\n  ],\n  \"optional_headers\": [\n    \"authorization\"\n  ],\n  \"methods\": [\n    \"health\",\n    \"chat\",\n    \"streamChat\",\n    \"streamChatWs\"\n  ],\n  \"endpoints\": [\n    \"/api/{}/health\",\n    \"/api/{}/chat\",\n    \"/api/{}/chat/stream\",\n    \"/api/{}/chat/ws\"\n  ],\n  \"streaming\": {{\n    \"sse\": {{\n      \"events\": [\n        \"token\",\n        \"done\"\n      ]\n    }},\n    \"websocket\": {{\n      \"events\": [\n        \"token\",\n        \"done\"\n      ]\n    }}\n  }}\n}}\n",
        api_version, api_version, api_version, api_version, api_version
    )
}

fn render_conversation_schema_json() -> String {
    "{\n  \"$schema\": \"https://json-schema.org/draft/2020-12/schema\",\n  \"title\": \"EnkaiConversationStateV1\",\n  \"type\": \"object\",\n  \"required\": [\n    \"schema_version\",\n    \"id\",\n    \"source\",\n    \"updated_ms\"\n  ],\n  \"properties\": {\n    \"schema_version\": {\n      \"const\": 1\n    },\n    \"id\": {\n      \"type\": \"string\",\n      \"minLength\": 1\n    },\n    \"messages\": {\n      \"type\": \"array\",\n      \"minItems\": 1,\n      \"items\": {\n        \"type\": \"object\",\n        \"required\": [\n          \"role\",\n          \"content\"\n        ],\n        \"properties\": {\n          \"role\": {\n            \"type\": \"string\",\n            \"enum\": [\n              \"user\",\n              \"assistant\"\n            ]\n          },\n          \"content\": {\n            \"type\": \"string\"\n          }\n        },\n        \"additionalProperties\": true\n      }\n    },\n    \"source\": {\n      \"type\": \"string\"\n    },\n    \"updated_ms\": {\n      \"type\": \"integer\"\n    },\n    \"user_text\": {\n      \"type\": \"string\"\n    },\n    \"reply\": {\n      \"type\": \"string\"\n    }\n  },\n  \"additionalProperties\": true\n}\n".to_string()
}

fn render_deploy_env_snapshot(api_version: &str, profile: BackendProfile) -> String {
    let mut required = vec![
        "ENKAI_APP_PROFILE",
        "ENKAI_API_VERSION",
        "ENKAI_SERVE_HOST",
        "ENKAI_SERVE_PORT",
        "ENKAI_CONVERSATION_DIR",
        "ENKAI_CONVERSATION_DB",
    ];
    if profile == BackendProfile::Llm {
        required.push("ENKAI_MODEL_NAME");
        required.push("ENKAI_MODEL_VERSION");
    }
    let required_json = required
        .into_iter()
        .map(|item| format!("    \"{}\"", item))
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        "{{\n  \"snapshot_version\": 1,\n  \"api_version\": \"{}\",\n  \"profile\": \"{}\",\n  \"required_env\": [\n{}\n  ],\n  \"optional_env\": [\n    \"ENKAI_MODEL_REGISTRY\",\n    \"ENKAI_API_TOKEN\",\n    \"ENKAI_LOG_PATH\"\n  ],\n  \"constraints\": {{\n    \"ENKAI_SERVE_PORT\": \"integer 1..65535\",\n    \"ENKAI_API_VERSION\": \"must equal scaffold api version\"\n  }}\n}}\n",
        api_version,
        backend_profile_slug(profile),
        required_json
    )
}

fn render_backend_env_validator_py(api_version: &str, profile: BackendProfile) -> String {
    let profile_slug = backend_profile_slug(profile);
    let mut required = vec![
        "ENKAI_APP_PROFILE",
        "ENKAI_API_VERSION",
        "ENKAI_SERVE_HOST",
        "ENKAI_SERVE_PORT",
        "ENKAI_CONVERSATION_DIR",
        "ENKAI_CONVERSATION_DB",
    ];
    if profile == BackendProfile::Llm {
        required.push("ENKAI_MODEL_NAME");
        required.push("ENKAI_MODEL_VERSION");
    }
    let required_list = required
        .into_iter()
        .map(|key| format!("\"{}\"", key))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "import argparse\nimport os\nimport pathlib\nimport sys\n\nEXPECTED_API_VERSION = \"{}\"\nEXPECTED_PROFILE = \"{}\"\nREQUIRED = [{}]\n\n\ndef parse_env_file(path):\n    values = {{}}\n    data = pathlib.Path(path)\n    if not data.exists():\n        return values\n    for line in data.read_text(encoding=\"utf-8\").splitlines():\n        raw = line.strip()\n        if not raw or raw.startswith(\"#\"):\n            continue\n        if \"=\" not in raw:\n            continue\n        key, value = raw.split(\"=\", 1)\n        values[key.strip()] = value.strip()\n    return values\n\n\ndef merged_env(file_values):\n    out = dict(file_values)\n    for key, value in os.environ.items():\n        out[key] = value\n    return out\n\n\ndef main():\n    parser = argparse.ArgumentParser(description=\"Validate Enkai deploy env contract\")\n    parser.add_argument(\"--env-file\", default=\".env\", help=\"Path to env file\")\n    args = parser.parse_args()\n\n    file_values = parse_env_file(args.env_file)\n    env = merged_env(file_values)\n\n    errors = []\n    for key in REQUIRED:\n        value = env.get(key, \"\").strip()\n        if not value:\n            errors.append(f\"missing required env: {{key}}\")\n\n    api_version = env.get(\"ENKAI_API_VERSION\", \"\").strip()\n    if api_version and api_version != EXPECTED_API_VERSION:\n        errors.append(\n            f\"ENKAI_API_VERSION mismatch: expected {{EXPECTED_API_VERSION}}, got {{api_version}}\"\n        )\n\n    profile = env.get(\"ENKAI_APP_PROFILE\", \"\").strip()\n    if profile and profile != EXPECTED_PROFILE:\n        errors.append(\n            f\"ENKAI_APP_PROFILE mismatch: expected {{EXPECTED_PROFILE}}, got {{profile}}\"\n        )\n\n    raw_port = env.get(\"ENKAI_SERVE_PORT\", \"\").strip()\n    if raw_port:\n        try:\n            port = int(raw_port)\n            if port < 1 or port > 65535:\n                errors.append(\"ENKAI_SERVE_PORT must be in range 1..65535\")\n        except ValueError:\n            errors.append(\"ENKAI_SERVE_PORT must be an integer\")\n\n    if errors:\n        for error in errors:\n            print(f\"[env-contract] {{error}}\", file=sys.stderr)\n        return 1\n\n    print(\"[env-contract] ok\")\n    return 0\n\n\nif __name__ == \"__main__\":\n    raise SystemExit(main())\n",
        api_version, profile_slug, required_list
    )
}

fn render_backend_main(api_version: &str) -> String {
    let base = format!("/api/{}", api_version);
    let health = format!("{}/health", base);
    let chat = format!("{}/chat", base);
    let stream = format!("{}/chat/stream", base);
    let ws = format!("{}/chat/ws", base);
    format!(
        "import std::http\n\
import std::io\n\
import std::env\n\
import std::path\n\
import std::time\n\n\
policy default ::\n\
    allow net\n\
    allow fs\n\
    allow env\n\
::\n\n\
fn conversation_dir() -> String ::\n\
    let dir := env.get(\"ENKAI_CONVERSATION_DIR\")\n\
    if dir == none ::\n\
        return \".\"\n\
    ::\n\
    return dir?\n\
::\n\n\
fn conversation_path() -> String ::\n\
    return path.join(conversation_dir(), \"conversation_state.json\")\n\
::\n\n\
fn conversation_backup_path() -> String ::\n\
    return path.join(conversation_dir(), \"conversation_state.backup.json\")\n\
::\n\n\
fn log_path() -> String ::\n\
    let value := env.get(\"ENKAI_LOG_PATH\")\n\
    if value != none ::\n\
        return value?\n\
    ::\n\
    return \"server.jsonl\"\n\
::\n\n\
fn env_api_version() -> String ::\n\
    let value := env.get(\"ENKAI_API_VERSION\")\n\
    if value == none ::\n\
        return \"{}\"\n\
    ::\n\
    return value?\n\
::\n\n\
fn env_port() -> Int ::\n\
    let value := env.get(\"ENKAI_SERVE_PORT\")\n\
    if value == none ::\n\
        return 8080\n\
    ::\n\
    let parsed := json.parse(value?)\n\
    if parsed < 1 ::\n\
        return 8080\n\
    ::\n\
    if parsed > 65535 ::\n\
        return 8080\n\
    ::\n\
    return parsed\n\
::\n\n\
fn contract_error(code: String, message: String) -> Response ::\n\
    let out := json.parse(\"{{}}\")\n\
    let err := json.parse(\"{{}}\")\n\
    err.code := code\n\
    err.message := message\n\
    err.api_version := \"{}\"\n\
    out.error := err\n\
    return http.bad_request(json.stringify(out))\n\
::\n\n\
fn request_api_version(req: Request) -> String? ::\n\
    return http.header(req, \"x-enkai-api-version\")\n\
::\n\n\
fn request_ws_api_version(req: Request) -> String? ::\n\
    let value := http.header(req, \"x-enkai-api-version\")\n\
    if value == none ::\n\
        value := http.query(req, \"api_version\")\n\
    ::\n\
    return value\n\
::\n\n\
fn next_conversation_id(value: String?) -> String ::\n\
    if value != none ::\n\
        return value?\n\
    ::\n\
    return json.stringify(time.now_ms())\n\
::\n\n\
fn migrate_conversation_state_file() ::\n\
    let raw := io.read_text(conversation_path())\n\
    if raw == none ::\n\
        raw := io.read_text(conversation_backup_path())\n\
    ::\n\
    if raw == none ::\n\
        return\n\
    ::\n\
    let parsed := json.parse(raw?)\n\
    parsed.schema_version := 1\n\
    io.write_text(conversation_path(), json.stringify(parsed))\n\
    io.write_text(conversation_backup_path(), json.stringify(parsed))\n\
::\n\n\
fn save_conversation(id: String, user_text: String, reply: String, source: String) ::\n\
    let user_msg := json.parse(\"{{}}\")\n\
    user_msg.role := \"user\"\n\
    user_msg.content := user_text\n\
    let assistant_msg := json.parse(\"{{}}\")\n\
    assistant_msg.role := \"assistant\"\n\
    assistant_msg.content := reply\n\
    let state := json.parse(\"{{}}\")\n\
    state.schema_version := 1\n\
    state.id := id\n\
    state.messages := [user_msg, assistant_msg]\n\
    state.user_text := user_text\n\
    state.reply := reply\n\
    state.source := source\n\
    state.updated_ms := time.now_ms()\n\
    io.write_text(conversation_path(), json.stringify(state))\n\
    io.write_text(conversation_backup_path(), json.stringify(state))\n\
::\n\n\
fn chat_payload(id: String, reply: String) -> String ::\n\
    let body := json.parse(\"{{}}\")\n\
    body.id := id\n\
    body.reply := reply\n\
    return json.stringify(body)\n\
::\n\n\
fn health(req: Request) -> Response ::\n\
    let version := request_api_version(req)\n\
    if version == none ::\n\
        return contract_error(\"missing_api_version_header\", \"missing x-enkai-api-version header\")\n\
    ::\n\
    if version? != \"{}\" ::\n\
        return contract_error(\"api_version_mismatch\", \"x-enkai-api-version mismatch\")\n\
    ::\n\
    return http.ok(\"{{\\\"status\\\":\\\"ok\\\",\\\"api_version\\\":\\\"{}\\\"}}\")\n\
::\n\n\
fn chat(req: Request) -> Response ::\n\
    let version := request_api_version(req)\n\
    if version == none ::\n\
        return contract_error(\"missing_api_version_header\", \"missing x-enkai-api-version header\")\n\
    ::\n\
    if version? != \"{}\" ::\n\
        return contract_error(\"api_version_mismatch\", \"x-enkai-api-version mismatch\")\n\
    ::\n\
    let prompt_query := http.query(req, \"prompt\")\n\
    if prompt_query == none ::\n\
        return contract_error(\"missing_prompt\", \"prompt query parameter is required\")\n\
    ::\n\
    let user_text := prompt_query?\n\
    let conversation_id := next_conversation_id(http.query(req, \"conversation_id\"))\n\
    let reply := \"hello from enkai backend\"\n\
    save_conversation(conversation_id, user_text, reply, \"chat\")\n\
    return http.ok(chat_payload(conversation_id, reply))\n\
::\n\n\
fn stream(req: Request) -> Response ::\n\
    let version := request_api_version(req)\n\
    if version == none ::\n\
        return contract_error(\"missing_api_version_header\", \"missing x-enkai-api-version header\")\n\
    ::\n\
    if version? != \"{}\" ::\n\
        return contract_error(\"api_version_mismatch\", \"x-enkai-api-version mismatch\")\n\
    ::\n\
    let prompt_query := http.query(req, \"prompt\")\n\
    if prompt_query == none ::\n\
        return contract_error(\"missing_prompt\", \"prompt query parameter is required\")\n\
    ::\n\
    let user_text := prompt_query?\n\
    let conversation_id := next_conversation_id(http.query(req, \"conversation_id\"))\n\
    let s := http.sse_open()\n\
    let token0 := json.parse(\"{{}}\")\n\
    token0.event := \"token\"\n\
    token0.value := \"hello\"\n\
    http.sse_send(s, json.stringify(token0))\n\
    let token1 := json.parse(\"{{}}\")\n\
    token1.event := \"token\"\n\
    token1.value := \" from\"\n\
    http.sse_send(s, json.stringify(token1))\n\
    let token2 := json.parse(\"{{}}\")\n\
    token2.event := \"token\"\n\
    token2.value := \" enkai\"\n\
    http.sse_send(s, json.stringify(token2))\n\
    let done := json.parse(\"{{}}\")\n\
    done.event := \"done\"\n\
    done.conversation_id := conversation_id\n\
    http.sse_send(s, json.stringify(done))\n\
    http.sse_close(s)\n\
    save_conversation(conversation_id, user_text, \"hello from enkai\", \"stream\")\n\
    return http.ok(\"\")\n\
::\n\n\
fn chat_ws(req: Request) -> Response ::\n\
    let version := request_ws_api_version(req)\n\
    if version == none ::\n\
        return contract_error(\"missing_api_version_header\", \"missing x-enkai-api-version header\")\n\
    ::\n\
    if version? != \"{}\" ::\n\
        return contract_error(\"api_version_mismatch\", \"x-enkai-api-version mismatch\")\n\
    ::\n\
    let prompt_query := http.query(req, \"prompt\")\n\
    if prompt_query == none ::\n\
        return contract_error(\"missing_prompt\", \"prompt query parameter is required\")\n\
    ::\n\
    let user_text := prompt_query?\n\
    let conversation_id := next_conversation_id(http.query(req, \"conversation_id\"))\n\
    let ws := http.ws_open(req)\n\
    let token0 := json.parse(\"{{}}\")\n\
    token0.event := \"token\"\n\
    token0.value := \"hello\"\n\
    http.ws_send(ws, json.stringify(token0))\n\
    let token1 := json.parse(\"{{}}\")\n\
    token1.event := \"token\"\n\
    token1.value := \" from\"\n\
    http.ws_send(ws, json.stringify(token1))\n\
    let token2 := json.parse(\"{{}}\")\n\
    token2.event := \"token\"\n\
    token2.value := \" enkai\"\n\
    http.ws_send(ws, json.stringify(token2))\n\
    let done := json.parse(\"{{}}\")\n\
    done.event := \"done\"\n\
    done.conversation_id := conversation_id\n\
    http.ws_send(ws, json.stringify(done))\n\
    http.ws_close(ws)\n\
    save_conversation(conversation_id, user_text, \"hello from enkai\", \"ws\")\n\
    return none\n\
::\n\n\
fn not_found(req: Request) -> Response ::\n\
    let out := json.parse(\"{{}}\")\n\
    let err := json.parse(\"{{}}\")\n\
    err.code := \"not_found\"\n\
    err.message := \"route not found\"\n\
    err.api_version := \"{}\"\n\
    out.error := err\n\
    return http.not_found(json.stringify(out))\n\
::\n\n\
fn main() ::\n\
    migrate_conversation_state_file()\n\
    if env_api_version() != \"{}\" ::\n\
        io.stderr_write_text(\"ENKAI_API_VERSION mismatch with scaffold contract\")\n\
        return\n\
    ::\n\
    let routes := [\n\
        http.route(\"GET\", \"{}\", health),\n\
        http.route(\"POST\", \"{}\", chat),\n\
        http.route(\"GET\", \"{}\", stream),\n\
        http.route(\"GET\", \"{}\", chat_ws)\n\
    ]\n\
    let auth_cfg := json.parse(\"{{\\\"allow_anonymous\\\":true,\\\"tokens\\\":[{{\\\"token\\\":\\\"dev-token\\\",\\\"tenant\\\":\\\"local\\\"}}]}}\")\n\
    let rate_cfg := json.parse(\"{{\\\"capacity\\\":120,\\\"refill_per_sec\\\":60,\\\"key\\\":\\\"ip\\\"}}\")\n\
    let log_cfg := json.parse(\"{{}}\")\n\
    log_cfg.path := log_path()\n\
    let middlewares := [\n\
        http.middleware(\"auth\", auth_cfg),\n\
        http.middleware(\"rate_limit\", rate_cfg),\n\
        http.middleware(\"jsonl_log\", log_cfg),\n\
        http.middleware(\"default\", not_found)\n\
    ]\n\
    let host := \"0.0.0.0\"\n\
    let host_env := env.get(\"ENKAI_SERVE_HOST\")\n\
    if host_env != none ::\n\
        host := host_env?\n\
    ::\n\
    let port := env_port()\n\
    http.serve(host, port, routes, middlewares)\n\
    let contract_mode := env.get(\"ENKAI_CONTRACT_TEST_MODE\")\n\
    if contract_mode != none ::\n\
        let until_ms := time.now_ms() + 3000\n\
        while time.now_ms() < until_ms ::\n\
            let tick_ms := time.now_ms()\n\
        ::\n\
        return\n\
    ::\n\
    while true ::\n\
        let tick_ms := time.now_ms()\n\
    ::\n\
::\n\
\n\
main()\n",
        api_version,
        api_version,
        api_version,
        api_version,
        api_version,
        api_version,
        api_version,
        api_version,
        api_version,
        health,
        chat,
        stream,
        ws
    )
}

fn render_frontend_package_json(root: &Path) -> String {
    let name = sanitize_name(root);
    format!(
        "{{\n  \"name\": \"{}\",\n  \"private\": true,\n  \"version\": \"0.1.0\",\n  \"type\": \"module\",\n  \"scripts\": {{\n    \"dev\": \"vite\",\n    \"build\": \"tsc -b && vite build\",\n    \"preview\": \"vite preview\"\n  }},\n  \"dependencies\": {{\n    \"react\": \"^18.3.1\",\n    \"react-dom\": \"^18.3.1\"\n  }},\n  \"devDependencies\": {{\n    \"@types/react\": \"^18.3.10\",\n    \"@types/react-dom\": \"^18.3.0\",\n    \"@vitejs/plugin-react\": \"^4.3.1\",\n    \"typescript\": \"^5.6.3\",\n    \"vite\": \"^5.4.8\"\n  }}\n}}\n",
        name
    )
}

fn render_frontend_env_example(api_version: &str, backend_url: &str) -> String {
    format!(
        "VITE_ENKAI_API_BASE_URL={}\nVITE_ENKAI_API_VERSION={}\nVITE_ENKAI_API_TOKEN=dev-token\n",
        backend_url, api_version
    )
}

fn render_frontend_readme(api_version: &str, backend_url: &str) -> String {
    format!(
        "# Frontend Chat Scaffold\n\nTyped frontend scaffold for Enkai serving APIs.\n\n## Contract\n\n- Base URL: `{}` (override via `.env.local`)\n- API version: `{}`\n- Header pinning: `x-enkai-api-version: {}`\n- Streaming transports: SSE (`/chat/stream`) and WebSocket (`/chat/ws`)\n\n## Run\n\n1. `npm install`\n2. `npm run dev`\n\n## SDK + Snapshot\n\n- Generated client: `src/sdk/enkaiClient.ts`\n- Contract snapshot: `contracts/sdk_api.snapshot.json`\n",
        backend_url, api_version, api_version
    )
}

fn render_frontend_app_tsx(api_version: &str) -> String {
    format!(
        "import {{ FormEvent, useMemo, useState }} from \"react\";\n\
import {{ EnkaiClient, ChatMessage }} from \"./sdk/enkaiClient\";\n\
\n\
type Role = \"user\" | \"assistant\";\n\
\n\
interface Message extends ChatMessage {{\n\
  id: string;\n\
  role: Role;\n\
}}\n\
\n\
const initialMessages: Message[] = [\n\
  {{ id: \"welcome\", role: \"assistant\", content: \"Enkai frontend scaffold ready.\" }}\n\
];\n\
\n\
function uid() {{\n\
  return Math.random().toString(36).slice(2);\n\
}}\n\
\n\
export default function App() {{\n\
  const [token, setToken] = useState(import.meta.env.VITE_ENKAI_API_TOKEN ?? \"\");\n\
  const [input, setInput] = useState(\"\");\n\
  const [conversationId, setConversationId] = useState<string | null>(null);\n\
  const [messages, setMessages] = useState<Message[]>(initialMessages);\n\
  const [streaming, setStreaming] = useState(false);\n\
  const [error, setError] = useState<string | null>(null);\n\
\n\
  const client = useMemo(() => new EnkaiClient({{\n\
    baseUrl: import.meta.env.VITE_ENKAI_API_BASE_URL ?? \"http://localhost:8080\",\n\
    apiVersion: import.meta.env.VITE_ENKAI_API_VERSION ?? \"{}\",\n\
    token: token || undefined,\n\
  }}), [token]);\n\
\n\
  async function onSubmit(event: FormEvent) {{\n\
    event.preventDefault();\n\
    const prompt = input.trim();\n\
    if (!prompt || streaming) {{\n\
      return;\n\
    }}\n\
    setError(null);\n\
    setInput(\"\");\n\
    const userMessage: Message = {{ id: uid(), role: \"user\", content: prompt }};\n\
    const assistantId = uid();\n\
    setMessages((prev) => [...prev, userMessage, {{ id: assistantId, role: \"assistant\", content: \"\" }}]);\n\
    setStreaming(true);\n\
    try {{\n\
      await client.streamChat(prompt, (eventData) => {{\n\
        if (eventData.event === \"token\" && eventData.value) {{\n\
          setMessages((prev) => prev.map((item) => item.id === assistantId\n\
            ? {{ ...item, content: item.content + eventData.value }}\n\
            : item\n\
          ));\n\
        }}\n\
        if (eventData.event === \"done\" && eventData.conversation_id) {{\n\
          setConversationId(eventData.conversation_id);\n\
        }}\n\
      }}, {{ conversationId: conversationId ?? undefined }});\n\
    }} catch (err) {{\n\
      const text = err instanceof Error ? err.message : \"stream failed\";\n\
      setError(text);\n\
      setMessages((prev) => prev.map((item) => item.id === assistantId\n\
        ? {{ ...item, content: \"[error] \" + text }}\n\
        : item\n\
      ));\n\
    }} finally {{\n\
      setStreaming(false);\n\
    }}\n\
  }}\n\
\n\
  return (\n\
    <div className=\"app-shell\">\n\
      <header className=\"hero\">\n\
        <p className=\"eyebrow\">Enkai v2.2.0 frontend contract freeze</p>\n\
        <h1>Streaming Chat UI Kit</h1>\n\
        <p className=\"subtitle\">Typed SDK, version-pinned API contract, and resilient error UX.</p>\n\
        <p className=\"subtitle\">{{conversationId ? `Conversation: ${{conversationId}}` : \"Conversation: new\"}}</p>\n\
      </header>\n\
\n\
      <section className=\"panel auth-panel\">\n\
        <label htmlFor=\"token\">Auth Token</label>\n\
        <input\n\
          id=\"token\"\n\
          value={{token}}\n\
          onChange={{(e) => setToken(e.target.value)}}\n\
          placeholder=\"Bearer token (optional)\"\n\
        />\n\
      </section>\n\
\n\
      <section className=\"panel transcript\" aria-live=\"polite\">\n\
        {{messages.map((message) => (\n\
          <article key={{message.id}} className={{\"bubble \" + message.role}}>\n\
            <span className=\"role\">{{message.role}}</span>\n\
            <p>{{message.content || (streaming && message.role === \"assistant\" ? \"...\" : \"\")}}</p>\n\
          </article>\n\
        ))}}\n\
      </section>\n\
\n\
      <form className=\"panel composer\" onSubmit={{onSubmit}}>\n\
        <input\n\
          value={{input}}\n\
          onChange={{(e) => setInput(e.target.value)}}\n\
          placeholder=\"Send a prompt\"\n\
        />\n\
        <button type=\"submit\" disabled={{streaming}}>\n\
          {{streaming ? \"Streaming...\" : \"Send\"}}\n\
        </button>\n\
      </form>\n\
\n\
      {{error ? <p className=\"error\">{{error}}</p> : null}}\n\
    </div>\n\
  );\n\
}}\n",
        api_version
    )
}

fn render_typescript_sdk(api_version: &str) -> String {
    format!(
        "export type Role = \"user\" | \"assistant\";\n\
\n\
export interface ChatMessage {{\n\
  role: Role;\n\
  content: string;\n\
}}\n\
\n\
export interface ChatRequest {{\n\
  prompt: string;\n\
  conversationId?: string;\n\
}}\n\
\n\
export interface ChatResponse {{\n\
  id: string;\n\
  reply: string;\n\
}}\n\
\n\
export interface HealthResponse {{\n\
  status: string;\n\
  api_version: string;\n\
}}\n\
\n\
export interface StreamEvent {{\n\
  event: string;\n\
  value?: string;\n\
  conversation_id?: string;\n\
}}\n\
\n\
export interface ApiErrorBody {{\n\
  error?: {{\n\
    code?: string;\n\
    message?: string;\n\
    api_version?: string;\n\
  }};\n\
}}\n\
\n\
export interface StreamOptions {{\n\
  conversationId?: string;\n\
  signal?: AbortSignal;\n\
}}\n\
\n\
export interface EnkaiClientConfig {{\n\
  baseUrl: string;\n\
  apiVersion?: string;\n\
  token?: string;\n\
  fetchImpl?: typeof fetch;\n\
}}\n\
\n\
export class EnkaiClient {{\n\
  private readonly config: Required<Pick<EnkaiClientConfig, \"baseUrl\" | \"apiVersion\">> & Omit<EnkaiClientConfig, \"baseUrl\" | \"apiVersion\">;\n\
  private readonly fetchImpl: typeof fetch;\n\
\n\
  constructor(config: EnkaiClientConfig) {{\n\
    const apiVersion = config.apiVersion ?? \"{}\";\n\
    this.config = {{\n\
      ...config,\n\
      baseUrl: trimSlash(config.baseUrl),\n\
      apiVersion,\n\
    }};\n\
    this.fetchImpl = config.fetchImpl ?? fetch;\n\
  }}\n\
\n\
  async health(signal?: AbortSignal): Promise<HealthResponse> {{\n\
    const response = await this.fetchImpl(this.endpoint(\"/health\"), {{\n\
      method: \"GET\",\n\
      headers: this.headers(),\n\
      signal,\n\
    }});\n\
    return this.readJson<HealthResponse>(response);\n\
  }}\n\
\n\
  async chat(input: ChatRequest, signal?: AbortSignal): Promise<ChatResponse> {{\n\
    const url = new URL(this.endpoint(\"/chat\"));\n\
    url.searchParams.set(\"prompt\", input.prompt);\n\
    if (input.conversationId) {{\n\
      url.searchParams.set(\"conversation_id\", input.conversationId);\n\
    }}\n\
    const response = await this.fetchImpl(url.toString(), {{\n\
      method: \"POST\",\n\
      headers: this.headers({{ \"content-type\": \"application/json\" }}),\n\
      body: JSON.stringify(input),\n\
      signal,\n\
    }});\n\
    return this.readJson<ChatResponse>(response);\n\
  }}\n\
\n\
  async streamChat(\n\
    prompt: string,\n\
    onEvent: (event: StreamEvent) => void,\n\
    options: StreamOptions = {{}},\n\
  ): Promise<void> {{\n\
    const url = new URL(this.endpoint(\"/chat/stream\"));\n\
    url.searchParams.set(\"prompt\", prompt);\n\
    if (options.conversationId) {{\n\
      url.searchParams.set(\"conversation_id\", options.conversationId);\n\
    }}\n\
    const response = await this.fetchImpl(url.toString(), {{\n\
      method: \"GET\",\n\
      headers: this.headers(),\n\
      signal: options.signal,\n\
    }});\n\
    if (!response.ok) {{\n\
      throw new Error(await this.readErrorDetail(response));\n\
    }}\n\
    if (!response.body) {{\n\
      throw new Error(\"streaming is unavailable: response body missing\");\n\
    }}\n\
    const reader = response.body.getReader();\n\
    const decoder = new TextDecoder();\n\
    let buffer = \"\";\n\
    while (true) {{\n\
      const result = await reader.read();\n\
      if (result.done) {{\n\
        break;\n\
      }}\n\
      buffer += decoder.decode(result.value, {{ stream: true }});\n\
      let marker = buffer.indexOf(\"\\n\\n\");\n\
      while (marker >= 0) {{\n\
        const block = buffer.slice(0, marker);\n\
        buffer = buffer.slice(marker + 2);\n\
        for (const line of block.split(\"\\n\")) {{\n\
          const trimmed = line.trim();\n\
          if (!trimmed.startsWith(\"data:\")) {{\n\
            continue;\n\
          }}\n\
          const payload = trimmed.slice(5).trim();\n\
          if (!payload) {{\n\
            continue;\n\
          }}\n\
          try {{\n\
            const event = JSON.parse(payload) as StreamEvent;\n\
            onEvent(event);\n\
          }} catch (err) {{\n\
            throw new Error(`invalid stream event payload: ${{payload}}`);\n\
          }}\n\
        }}\n\
        marker = buffer.indexOf(\"\\n\\n\");\n\
      }}\n\
    }}\n\
  }}\n\
\n\
  streamChatWs(\n\
    prompt: string,\n\
    onEvent: (event: StreamEvent) => void,\n\
    options: StreamOptions = {{}},\n\
  ): () => void {{\n\
    const url = new URL(this.endpoint(\"/chat/ws\"));\n\
    url.searchParams.set(\"prompt\", prompt);\n\
    if (options.conversationId) {{\n\
      url.searchParams.set(\"conversation_id\", options.conversationId);\n\
    }}\n\
    url.searchParams.set(\"api_version\", this.config.apiVersion);\n\
    url.protocol = url.protocol === \"https:\" ? \"wss:\" : \"ws:\";\n\
    const socket = new WebSocket(url.toString());\n\
    socket.onmessage = (event) => {{\n\
      try {{\n\
        const payload = JSON.parse(String(event.data)) as StreamEvent;\n\
        onEvent(payload);\n\
      }} catch (_err) {{\n\
        // Ignore malformed messages from non-contract servers.\n\
      }}\n\
    }};\n\
    return () => socket.close();\n\
  }}\n\
\n\
  private endpoint(path: string): string {{\n\
    return `${{this.config.baseUrl}}/api/${{this.config.apiVersion}}${{path}}`;\n\
  }}\n\
\n\
  private headers(extra: Record<string, string> = {{}}): HeadersInit {{\n\
    const out: Record<string, string> = {{\n\
      \"x-enkai-api-version\": this.config.apiVersion,\n\
      ...extra,\n\
    }};\n\
    if (this.config.token) {{\n\
      out.authorization = `Bearer ${{this.config.token}}`;\n\
    }}\n\
    return out;\n\
  }}\n\
\n\
  private async readJson<T>(response: Response): Promise<T> {{\n\
    if (!response.ok) {{\n\
      throw new Error(await this.readErrorDetail(response));\n\
    }}\n\
    return (await response.json()) as T;\n\
  }}\n\
\n\
  private async readErrorDetail(response: Response): Promise<string> {{\n\
    let bodyText = \"\";\n\
    try {{\n\
      bodyText = await response.text();\n\
    }} catch (_err) {{\n\
      return `request failed: ${{response.status}}`;\n\
    }}\n\
    if (!bodyText) {{\n\
      return `request failed: ${{response.status}}`;\n\
    }}\n\
    try {{\n\
      const payload = JSON.parse(bodyText) as ApiErrorBody;\n\
      if (payload.error?.code && payload.error?.message) {{\n\
        return `request failed: ${{response.status}} (${{payload.error.code}}) ${{payload.error.message}}`;\n\
      }}\n\
      if (payload.error?.message) {{\n\
        return `request failed: ${{response.status}} ${{payload.error.message}}`;\n\
      }}\n\
    }} catch (_err) {{\n\
      // Fall through to generic output with raw body.\n\
    }}\n\
    return `request failed: ${{response.status}} ${{bodyText}}`;\n\
  }}\n\
}}\n\
\n\
function trimSlash(value: string): string {{\n\
  return value.endsWith(\"/\") ? value.slice(0, -1) : value;\n\
}}\n",
        api_version
    )
}

fn render_fullstack_readme(
    api_version: &str,
    backend_url: &str,
    backend_profile: BackendProfile,
) -> String {
    format!(
        "# Fullstack Chat Scaffold\n\nThis scaffold contains:\n\n- `backend/` Enkai serving project (`{}` profile)\n- `frontend/` React + TypeScript client app\n\n## API Contract\n\n- URL: `{}`\n- API version: `{}`\n- Pinned header: `x-enkai-api-version`\n- Streaming transports: SSE + WebSocket\n\n## Snapshots\n\n- `backend/contracts/backend_api.snapshot.json`\n- `backend/contracts/conversation_state.schema.json`\n- `backend/contracts/deploy_env.snapshot.json`\n- `frontend/contracts/sdk_api.snapshot.json`\n\n## Start\n\n1. Backend: `cd backend && enkai serve --host 0.0.0.0 --port 8080 .`\n2. Frontend: `cd frontend && npm install && npm run dev`\n",
        backend_profile_name(backend_profile),
        backend_url,
        api_version
    )
}

fn sanitize_name(path: &Path) -> String {
    let fallback = "enkai-app".to_string();
    let value = path
        .file_name()
        .and_then(|v| v.to_str())
        .map(|v| v.to_ascii_lowercase())
        .unwrap_or(fallback);
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else if ch.is_whitespace() || ch == '.' {
            out.push('-');
        }
    }
    if out.is_empty() {
        "enkai-app".to_string()
    } else {
        out
    }
}

const FRONTEND_INDEX_HTML: &str = "<!doctype html>\n<html lang=\"en\">\n  <head>\n    <meta charset=\"UTF-8\" />\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n    <title>Enkai Chat</title>\n  </head>\n  <body>\n    <div id=\"root\"></div>\n    <script type=\"module\" src=\"/src/main.tsx\"></script>\n  </body>\n</html>\n";

const FRONTEND_MAIN_TSX: &str = "import React from \"react\";\nimport ReactDOM from \"react-dom/client\";\nimport App from \"./App\";\nimport \"./styles.css\";\n\nReactDOM.createRoot(document.getElementById(\"root\")!).render(\n  <React.StrictMode>\n    <App />\n  </React.StrictMode>\n);\n";

const FRONTEND_TSCONFIG: &str = "{\n  \"compilerOptions\": {\n    \"target\": \"ES2022\",\n    \"module\": \"ESNext\",\n    \"moduleResolution\": \"Bundler\",\n    \"strict\": true,\n    \"jsx\": \"react-jsx\",\n    \"resolveJsonModule\": true,\n    \"isolatedModules\": true,\n    \"types\": [\"vite/client\"]\n  },\n  \"include\": [\"src\"]\n}\n";

const FRONTEND_TSCONFIG_NODE: &str = "{\n  \"compilerOptions\": {\n    \"composite\": true,\n    \"module\": \"ESNext\",\n    \"moduleResolution\": \"Bundler\",\n    \"types\": [\"node\"]\n  },\n  \"include\": [\"vite.config.ts\"]\n}\n";

const FRONTEND_VITE_CONFIG: &str = "import { defineConfig } from \"vite\";\nimport react from \"@vitejs/plugin-react\";\n\nexport default defineConfig({\n  plugins: [react()],\n});\n";

const FRONTEND_TYPES_TS: &str = "export type UiRole = \"user\" | \"assistant\";\n";

const MIGRATION_001_CONVERSATION_STATE_SQL: &str = "-- Enkai backend conversation persistence v1\nCREATE TABLE IF NOT EXISTS schema_migrations (\n  version INTEGER PRIMARY KEY,\n  applied_ms INTEGER NOT NULL\n);\n\nCREATE TABLE IF NOT EXISTS conversation_events (\n  id INTEGER PRIMARY KEY AUTOINCREMENT,\n  source_code INTEGER NOT NULL,\n  updated_ms INTEGER NOT NULL\n);\n";

const MIGRATION_002_CONVERSATION_STATE_INDEX_SQL: &str = "-- Enkai backend conversation persistence v2\nCREATE INDEX IF NOT EXISTS idx_conversation_events_updated_ms ON conversation_events(updated_ms);\n";

const FRONTEND_STYLES_CSS: &str = "@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');\n\n:root {\n  font-family: \"Space Grotesk\", \"Segoe UI\", sans-serif;\n  color: #f7f6f2;\n  background: radial-gradient(circle at 15% 10%, #233a66, transparent 42%), radial-gradient(circle at 80% 0%, #b14922, transparent 36%), linear-gradient(180deg, #0a1628 0%, #12253f 48%, #0f1d34 100%);\n}\n\n* {\n  box-sizing: border-box;\n}\n\nbody {\n  margin: 0;\n  min-height: 100vh;\n}\n\n#root {\n  min-height: 100vh;\n  padding: 1.5rem;\n}\n\n.app-shell {\n  max-width: 880px;\n  margin: 0 auto;\n  display: grid;\n  gap: 1rem;\n}\n\n.hero h1 {\n  margin: 0;\n  font-size: clamp(1.7rem, 4vw, 2.4rem);\n}\n\n.eyebrow {\n  font-family: \"IBM Plex Mono\", monospace;\n  letter-spacing: 0.1em;\n  text-transform: uppercase;\n  opacity: 0.8;\n  margin: 0 0 0.4rem;\n}\n\n.subtitle {\n  margin: 0.4rem 0 0;\n  opacity: 0.85;\n}\n\n.panel {\n  border: 1px solid rgba(255, 255, 255, 0.2);\n  background: rgba(6, 16, 30, 0.55);\n  backdrop-filter: blur(12px);\n  border-radius: 16px;\n  padding: 0.9rem;\n}\n\n.auth-panel {\n  display: grid;\n  gap: 0.4rem;\n}\n\ninput,\nbutton {\n  border: 1px solid rgba(255, 255, 255, 0.24);\n  border-radius: 10px;\n  padding: 0.75rem;\n  font: inherit;\n  color: inherit;\n  background: rgba(10, 25, 46, 0.82);\n}\n\nbutton {\n  cursor: pointer;\n  background: linear-gradient(120deg, #e66a2f, #d35317);\n  border: none;\n  font-weight: 600;\n}\n\nbutton:disabled {\n  opacity: 0.6;\n  cursor: not-allowed;\n}\n\n.transcript {\n  display: grid;\n  gap: 0.7rem;\n  min-height: 300px;\n}\n\n.bubble {\n  border-radius: 12px;\n  padding: 0.75rem;\n  animation: rise-in 220ms ease;\n}\n\n.bubble.user {\n  background: rgba(57, 123, 200, 0.32);\n}\n\n.bubble.assistant {\n  background: rgba(237, 122, 66, 0.28);\n}\n\n.role {\n  display: block;\n  font-family: \"IBM Plex Mono\", monospace;\n  font-size: 0.72rem;\n  letter-spacing: 0.06em;\n  text-transform: uppercase;\n  opacity: 0.82;\n}\n\n.bubble p {\n  margin: 0.25rem 0 0;\n  white-space: pre-wrap;\n}\n\n.composer {\n  display: grid;\n  grid-template-columns: 1fr auto;\n  gap: 0.7rem;\n}\n\n.error {\n  margin: 0;\n  color: #ffd9c8;\n}\n\n@keyframes rise-in {\n  from {\n    opacity: 0;\n    transform: translateY(6px);\n  }\n  to {\n    opacity: 1;\n    transform: translateY(0);\n  }\n}\n\n@media (max-width: 640px) {\n  #root {\n    padding: 1rem;\n  }\n\n  .composer {\n    grid-template-columns: 1fr;\n  }\n}\n";

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::path::Path;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use enkai_compiler::compiler::compile_package;
    use enkai_compiler::modules::load_package;
    use enkai_runtime::VM;
    use serde_json::Value as JsonValue;

    use tempfile::tempdir;

    fn frontend_contract_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::env_guard()
    }

    fn send_http_request(host: &str, port: u16, request: &str) -> Vec<u8> {
        let mut stream = TcpStream::connect((host, port)).expect("connect");
        stream.write_all(request.as_bytes()).expect("write");
        let mut buf = Vec::new();
        let mut chunk = [0u8; 1024];
        loop {
            match stream.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
                Err(err) if err.kind() == std::io::ErrorKind::ConnectionReset => break,
                Err(err) => panic!("read: {}", err),
            }
        }
        buf
    }

    fn response_status(raw: &[u8]) -> u16 {
        let text = String::from_utf8_lossy(raw);
        let line = text.lines().next().unwrap_or_default();
        line.split_whitespace()
            .nth(1)
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(0)
    }

    fn response_body(raw: &[u8]) -> String {
        let text = String::from_utf8_lossy(raw);
        if let Some(idx) = text.find("\r\n\r\n") {
            text[idx + 4..].to_string()
        } else {
            String::new()
        }
    }

    fn extract_conversation_id(stream_body: &str) -> Option<String> {
        let marker = "\"conversation_id\":\"";
        let idx = stream_body.find(marker)?;
        let rest = &stream_body[idx + marker.len()..];
        let end = rest.find('"')?;
        Some(rest[..end].to_string())
    }

    fn normalize_newlines(value: &str) -> String {
        value.replace("\r\n", "\n")
    }

    #[test]
    fn sdk_template_contains_version_pinning_contract() {
        let sdk = render_typescript_sdk("v9");
        assert!(sdk.contains("x-enkai-api-version"));
        assert!(sdk.contains("this.endpoint(\"/health\")"));
        assert!(sdk.contains("/api/${this.config.apiVersion}${path}"));
        assert!(sdk.contains("conversation_id"));
        assert!(sdk.contains("conversationId"));
        assert!(sdk.contains("apiVersion ?? \"v9\""));
        assert!(sdk.contains("streamChatWs("));
        assert!(sdk.contains("this.endpoint(\"/chat/ws\")"));
        assert!(sdk.contains("readErrorDetail"));
        assert!(sdk.contains("api_version"));
    }

    #[test]
    fn backend_template_contains_expected_routes() {
        let text = render_backend_main("v4");
        assert!(text.contains("/api/v4/health"));
        assert!(text.contains("/api/v4/chat"));
        assert!(text.contains("/api/v4/chat/stream"));
        assert!(text.contains("/api/v4/chat/ws"));
        assert!(text.contains("http.middleware(\"auth\""));
        assert!(text.contains("http.middleware(\"rate_limit\""));
        assert!(text.contains("save_conversation"));
        assert!(text.contains("ENKAI_CONTRACT_TEST_MODE"));
        assert!(text.contains("request_api_version"));
        assert!(text.contains("request_ws_api_version"));
        assert!(text.contains("schema_version := 1"));
        assert!(text.contains("migrate_conversation_state_file"));
    }

    #[test]
    fn contract_snapshots_match_reference_files() {
        assert_eq!(
            normalize_newlines(&render_backend_contract_snapshot("v1")),
            normalize_newlines(include_str!("../contracts/backend_api_v1.snapshot.json"))
        );
        assert_eq!(
            normalize_newlines(&render_sdk_contract_snapshot("v1")),
            normalize_newlines(include_str!("../contracts/sdk_api_v1.snapshot.json"))
        );
        assert_eq!(
            normalize_newlines(&render_conversation_schema_json()),
            normalize_newlines(include_str!(
                "../contracts/conversation_state_v1.schema.json"
            ))
        );
        assert_eq!(
            normalize_newlines(&render_deploy_env_snapshot("v1", BackendProfile::Backend)),
            normalize_newlines(include_str!(
                "../contracts/deploy_env_backend_v1.snapshot.json"
            ))
        );
        assert_eq!(
            normalize_newlines(&render_deploy_env_snapshot("v1", BackendProfile::Llm)),
            normalize_newlines(include_str!("../contracts/deploy_env_llm_v1.snapshot.json"))
        );
    }

    #[test]
    fn new_command_creates_frontend_scaffold() {
        let dir = tempdir().expect("tempdir");
        let target = dir.path().join("frontend-chat");
        let code = new_command(&[
            "frontend-chat".to_string(),
            target.to_string_lossy().to_string(),
            "--api-version".to_string(),
            "v2".to_string(),
        ]);
        assert_eq!(code, 0);
        assert!(target.join("src").join("App.tsx").is_file());
        assert!(target
            .join("src")
            .join("sdk")
            .join("enkaiClient.ts")
            .is_file());
        let env = fs::read_to_string(target.join(".env.example")).expect("env");
        assert!(env.contains("VITE_ENKAI_API_VERSION=v2"));
    }

    #[test]
    fn new_command_creates_service_scaffold() {
        let dir = tempdir().expect("tempdir");
        let target = dir.path().join("service");
        let code = new_command(&[
            "service".to_string(),
            target.to_string_lossy().to_string(),
            "--api-version".to_string(),
            "v3".to_string(),
        ]);
        assert_eq!(code, 0);
        assert!(target
            .join("migrations")
            .join("001_conversation_state.sql")
            .is_file());
        assert!(target
            .join("scripts")
            .join("validate_env_contract.py")
            .is_file());
        assert!(target
            .join("contracts")
            .join("deploy_env.snapshot.json")
            .is_file());
        let env = fs::read_to_string(target.join(".env.example")).expect("env");
        assert!(env.contains("ENKAI_APP_PROFILE=service"));
        assert!(env.contains("ENKAI_API_VERSION=v3"));
    }

    #[test]
    fn new_command_creates_llm_fullstack_scaffold() {
        let dir = tempdir().expect("tempdir");
        let target = dir.path().join("llm-fullstack");
        let code = new_command(&[
            "llm-fullstack".to_string(),
            target.to_string_lossy().to_string(),
            "--api-version".to_string(),
            "v5".to_string(),
            "--backend-url".to_string(),
            "http://localhost:8085".to_string(),
        ]);
        assert_eq!(code, 0);
        let backend_env =
            fs::read_to_string(target.join("backend").join(".env.example")).expect("backend env");
        assert!(backend_env.contains("ENKAI_APP_PROFILE=llm"));
        assert!(backend_env.contains("ENKAI_MODEL_NAME="));
        let deploy_snapshot = fs::read_to_string(
            target
                .join("backend")
                .join("contracts")
                .join("deploy_env.snapshot.json"),
        )
        .expect("deploy snapshot");
        assert!(deploy_snapshot.contains("\"profile\": \"llm\""));
    }

    #[test]
    fn sdk_generate_writes_output_file() {
        let dir = tempdir().expect("tempdir");
        let output = dir.path().join("enkai-client.ts");
        let code = sdk_command(&[
            "generate".to_string(),
            output.to_string_lossy().to_string(),
            "--api-version".to_string(),
            "v3".to_string(),
        ]);
        assert_eq!(code, 0);
        let text = fs::read_to_string(output).expect("sdk output");
        assert!(text.contains("apiVersion ?? \"v3\""));
    }

    #[test]
    fn parse_new_kind_rejects_unknown() {
        let err = parse_new_kind("mobile-app").expect_err("invalid kind");
        assert!(err.contains("unknown scaffold kind"));
    }

    #[test]
    fn prepare_destination_rejects_non_empty_without_force() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("app");
        fs::create_dir_all(&path).expect("dir");
        fs::write(path.join("existing.txt"), "x").expect("file");
        let err = prepare_destination(&path, false).expect_err("should fail");
        assert!(err.contains("not empty"));
    }

    #[test]
    fn sanitize_name_normalizes_path() {
        let name = sanitize_name(Path::new("A Cool.App"));
        assert_eq!(name, "a-cool-app");
    }

    #[test]
    fn fullstack_stream_contract_and_persistence_across_versions() {
        let _guard = frontend_contract_guard();
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("fullstack");
        let code = new_command(&[
            "fullstack-chat".to_string(),
            root.to_string_lossy().to_string(),
            "--api-version".to_string(),
            "v1".to_string(),
            "--backend-url".to_string(),
            "http://127.0.0.1:8080".to_string(),
        ]);
        assert_eq!(code, 0);

        let backend_entry = root.join("backend").join("src").join("main.enk");
        assert!(backend_entry.is_file());
        let backend_snapshot = root
            .join("backend")
            .join("contracts")
            .join("backend_api.snapshot.json");
        assert!(backend_snapshot.is_file());
        assert_eq!(
            fs::read_to_string(&backend_snapshot).expect("backend snapshot"),
            render_backend_contract_snapshot("v1")
        );
        let conversation_schema = root
            .join("backend")
            .join("contracts")
            .join("conversation_state.schema.json");
        assert!(conversation_schema.is_file());
        assert_eq!(
            fs::read_to_string(&conversation_schema).expect("conversation schema"),
            render_conversation_schema_json()
        );
        let deploy_snapshot = root
            .join("backend")
            .join("contracts")
            .join("deploy_env.snapshot.json");
        assert!(deploy_snapshot.is_file());
        assert_eq!(
            fs::read_to_string(&deploy_snapshot).expect("deploy snapshot"),
            render_deploy_env_snapshot("v1", BackendProfile::Backend)
        );
        let frontend_sdk = root
            .join("frontend")
            .join("src")
            .join("sdk")
            .join("enkaiClient.ts");
        assert!(frontend_sdk.is_file());
        let sdk_snapshot = root
            .join("frontend")
            .join("contracts")
            .join("sdk_api.snapshot.json");
        assert!(sdk_snapshot.is_file());
        assert_eq!(
            fs::read_to_string(&sdk_snapshot).expect("sdk snapshot"),
            render_sdk_contract_snapshot("v1")
        );

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        drop(listener);

        let conversation_dir = root.join("backend").join("state");
        fs::create_dir_all(&conversation_dir).expect("conversation dir");
        let legacy_state = serde_json::json!({
            "id": "legacy-1",
            "user_text": "legacy user",
            "reply": "legacy reply",
            "source": "legacy",
            "updated_ms": 1
        });
        fs::write(
            conversation_dir.join("conversation_state.json"),
            serde_json::to_string(&legacy_state).expect("legacy json"),
        )
        .expect("write legacy state");

        let std_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .join("std");
        std::env::set_var("ENKAI_SERVE_HOST", "127.0.0.1");
        std::env::set_var("ENKAI_SERVE_PORT", port.to_string());
        std::env::set_var("ENKAI_CONTRACT_TEST_MODE", "1");
        std::env::set_var("ENKAI_STD", std_root.to_string_lossy().to_string());
        std::env::set_var(
            "ENKAI_CONVERSATION_DIR",
            conversation_dir.to_string_lossy().to_string(),
        );

        let package = load_package(&backend_entry).expect("load backend");
        let program = compile_package(&package).expect("compile backend");
        let (backend_done_tx, backend_done_rx) = mpsc::channel();
        let backend = std::thread::spawn(move || {
            let mut vm = VM::new(false, false, false, false);
            let result = vm.run(&program).map(|_| ()).map_err(|err| err.to_string());
            let _ = backend_done_tx.send(result);
        });
        let wait_start = Instant::now();
        loop {
            if TcpStream::connect(("127.0.0.1", port)).is_ok() {
                break;
            }
            if let Ok(result) = backend_done_rx.try_recv() {
                panic!("backend exited before server startup: {:?}", result);
            }
            if wait_start.elapsed() >= Duration::from_secs(2) {
                panic!("server did not start on 127.0.0.1:{} within 2s", port);
            }
            std::thread::sleep(Duration::from_millis(40));
        }

        let stream_request =
            "GET /api/v1/chat/stream?prompt=hello HTTP/1.1\r\nHost: localhost\r\nx-enkai-api-version: v1\r\nConnection: close\r\n\r\n"
                .to_string();
        let stream_resp = send_http_request("127.0.0.1", port, &stream_request);
        let stream_body = response_body(&stream_resp);
        assert_eq!(
            response_status(&stream_resp),
            200,
            "unexpected stream response body: {}",
            stream_body
        );
        assert!(
            stream_body.contains("\"event\":\"token\""),
            "stream body: {}",
            stream_body
        );
        let conversation_id = extract_conversation_id(&stream_body).expect("conversation_id");
        let chat_request = format!(
            "POST /api/v1/chat?prompt=resume&conversation_id={} HTTP/1.1\r\nHost: localhost\r\nx-enkai-api-version: v1\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",
            conversation_id
        );
        let chat_resp = send_http_request("127.0.0.1", port, &chat_request);
        let chat_body = response_body(&chat_resp);
        assert_eq!(
            response_status(&chat_resp),
            200,
            "unexpected chat response body: {}",
            chat_body
        );
        assert!(
            chat_body.contains(&conversation_id),
            "chat body: {}",
            chat_body
        );

        let mismatch_request =
            "GET /api/v2/chat/stream?prompt=hello HTTP/1.1\r\nHost: localhost\r\nx-enkai-api-version: v2\r\nConnection: close\r\n\r\n"
                .to_string();
        let mismatch_resp = send_http_request("127.0.0.1", port, &mismatch_request);
        assert_eq!(response_status(&mismatch_resp), 404);

        let missing_header_request =
            "POST /api/v1/chat?prompt=hello HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
                .to_string();
        let missing_header_resp = send_http_request("127.0.0.1", port, &missing_header_request);
        assert_eq!(response_status(&missing_header_resp), 400);
        let missing_header_body = response_body(&missing_header_resp);
        assert!(missing_header_body.contains("missing_api_version_header"));

        let upgraded_state_text =
            fs::read_to_string(conversation_dir.join("conversation_state.json"))
                .expect("upgraded conversation state");
        let upgraded_state: JsonValue =
            serde_json::from_str(&upgraded_state_text).expect("valid upgraded json");
        assert_eq!(
            upgraded_state
                .get("schema_version")
                .and_then(|v| v.as_u64()),
            Some(1)
        );
        assert!(upgraded_state
            .get("messages")
            .and_then(|v| v.as_array())
            .is_some());
        assert!(upgraded_state.get("id").and_then(|v| v.as_str()).is_some());

        backend.join().expect("backend join");
        std::env::remove_var("ENKAI_SERVE_HOST");
        std::env::remove_var("ENKAI_SERVE_PORT");
        std::env::remove_var("ENKAI_CONTRACT_TEST_MODE");
        std::env::remove_var("ENKAI_STD");
        std::env::remove_var("ENKAI_CONVERSATION_DIR");
    }

    #[test]
    fn fullstack_force_rescaffold_updates_contract_version() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("fullstack-upgrade");
        let code = new_command(&[
            "fullstack-chat".to_string(),
            root.to_string_lossy().to_string(),
            "--api-version".to_string(),
            "v1".to_string(),
            "--backend-url".to_string(),
            "http://127.0.0.1:8080".to_string(),
        ]);
        assert_eq!(code, 0);

        let code = new_command(&[
            "fullstack-chat".to_string(),
            root.to_string_lossy().to_string(),
            "--api-version".to_string(),
            "v2".to_string(),
            "--backend-url".to_string(),
            "http://127.0.0.1:8080".to_string(),
            "--force".to_string(),
        ]);
        assert_eq!(code, 0);

        let backend_snapshot = fs::read_to_string(
            root.join("backend")
                .join("contracts")
                .join("backend_api.snapshot.json"),
        )
        .expect("backend snapshot");
        assert!(backend_snapshot.contains("\"api_version\": \"v2\""));
        assert!(backend_snapshot.contains("/api/v2/chat/stream"));

        let sdk_snapshot = fs::read_to_string(
            root.join("frontend")
                .join("contracts")
                .join("sdk_api.snapshot.json"),
        )
        .expect("sdk snapshot");
        assert!(sdk_snapshot.contains("\"api_version\": \"v2\""));

        let frontend_env =
            fs::read_to_string(root.join("frontend").join(".env.example")).expect("frontend env");
        assert!(frontend_env.contains("VITE_ENKAI_API_VERSION=v2"));
    }
}
