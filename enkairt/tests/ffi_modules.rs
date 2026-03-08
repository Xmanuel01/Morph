use std::fs;
use std::path::{Path, PathBuf};

use enkaic::compiler::compile_package;
use enkaic::modules::load_package;
use enkairt::error::RuntimeError;
use enkairt::object::Obj;
use enkairt::{Value, VM};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .to_path_buf()
}

fn copy_std_modules(dest: &Path) {
    let std_src = repo_root().join("std");
    let std_dst = dest.join("std");
    fs::create_dir_all(&std_dst).expect("std dst");
    for entry in fs::read_dir(&std_src).expect("std entries") {
        let entry = entry.expect("std entry");
        let path = entry.path();
        if matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("enk") | Some("en") | Some("enkai")
        ) {
            let name = path.file_name().expect("name");
            fs::copy(&path, std_dst.join(name)).expect("copy std module");
        }
    }
}

const POLICY_ALLOW_ALL: &str =
    "policy default ::\n    allow io\n    allow fs\n    allow env\n    allow process\n    allow net\n    allow db\n::\n\n";

fn inject_policy(source: &str) -> String {
    if source.contains("policy ") {
        return source.to_string();
    }
    let mut insert_at = 0usize;
    let mut in_native = false;
    for (idx, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if in_native {
            if trimmed == "::" {
                in_native = false;
                insert_at = idx + 1;
            }
            continue;
        }
        if trimmed.starts_with("import ") {
            insert_at = idx + 1;
            continue;
        }
        if trimmed.starts_with("native::import ") {
            in_native = true;
            insert_at = idx + 1;
            continue;
        }
        break;
    }
    let mut out = String::new();
    let mut inserted = false;
    for (idx, line) in source.lines().enumerate() {
        if !inserted && idx == insert_at {
            out.push_str(POLICY_ALLOW_ALL);
            inserted = true;
        }
        out.push_str(line);
        out.push('\n');
    }
    if !inserted {
        out.push_str(POLICY_ALLOW_ALL);
    }
    out
}

fn run_package(root: &Path, entry_name: &str, source: &str) -> Result<Value, RuntimeError> {
    let entry = root.join(entry_name);
    let source = inject_policy(source);
    fs::write(&entry, source).expect("write entry");
    let package = load_package(&entry).expect("load package");
    let program = compile_package(&package).expect("compile");
    let mut vm = VM::new(false, false, false, false);
    vm.run(&program)
}

fn run_package_raw(root: &Path, entry_name: &str, source: &str) -> Result<Value, RuntimeError> {
    let entry = root.join(entry_name);
    fs::write(&entry, source).expect("write entry");
    let package = load_package(&entry).expect("load package");
    let program = compile_package(&package).expect("compile");
    let mut vm = VM::new(false, false, false, false);
    vm.run(&program)
}

#[test]
fn std_fsx_roundtrip() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let data_path = temp.path().join("data.bin");
    let data_path = data_path.to_string_lossy().replace('\\', "/");
    let source = format!(
        "import std::fsx\n\n\
        native::import \"enkai_native\" ::\n    fn buffer_from_string(data: String) -> Buffer\n    fn buffer_eq(a: Buffer, b: Buffer) -> Bool\n::\n\
        let path := \"{}\"\n\
        let buf := buffer_from_string(\"hello\")\n\
        fsx.write_bytes(path, buf)\n\
        let out := fsx.read_bytes(path)\n\
        buffer_eq(buf, out)\n",
        data_path
    );
    let value = run_package(temp.path(), "main.enk", &source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_zstd_roundtrip() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let source = "import std::zstd\n\
        native::import \"enkai_native\" ::\n    fn buffer_from_string(data: String) -> Buffer\n    fn buffer_eq(a: Buffer, b: Buffer) -> Bool\n::\n\
        let input := buffer_from_string(\"hello\")\n\
        let compressed := zstd.compress(input, 1)\n\
        let output := zstd.decompress(compressed)\n\
        buffer_eq(input, output)\n";
    let value = run_package(temp.path(), "main.enk", source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_hash_sha256_len() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let source = "import std::hash\n\
        native::import \"enkai_native\" ::\n    fn buffer_len(data: Buffer) -> Int\n::\n\
        let digest := hash.sha256_from_string(\"abc\")\n\
        buffer_len(digest)\n";
    let value = run_package(temp.path(), "main.enk", source).expect("run");
    assert_eq!(value, Value::Int(32));
}

#[test]
fn std_env_get_set_roundtrip() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let source = "import std::env\n\
        native::import \"enkai_native\" ::\n    fn buffer_from_string(data: String) -> Buffer\n    fn buffer_eq(a: Buffer, b: Buffer) -> Bool\n::\n\
        let ok := env.set(\"ENKAI_TEST_KEY\", \"hello\")\n\
        let val := env.get(\"ENKAI_TEST_KEY\")?\n\
        env.remove(\"ENKAI_TEST_KEY\")\n\
        let expected := buffer_from_string(\"hello\")\n\
        let actual := buffer_from_string(val)\n\
        ok and buffer_eq(expected, actual)\n";
    let value = run_package(temp.path(), "main.enk", source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_path_basename() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let source = "import std::path\n\
        native::import \"enkai_native\" ::\n    fn buffer_from_string(data: String) -> Buffer\n    fn buffer_eq(a: Buffer, b: Buffer) -> Bool\n::\n\
        let joined := path.join(\"foo\", \"bar\")\n\
        let base := path.basename(joined)?\n\
        let expected := buffer_from_string(\"bar\")\n\
        let actual := buffer_from_string(base)\n\
        buffer_eq(expected, actual)\n";
    let value = run_package(temp.path(), "main.enk", source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_time_now_ms() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let source = "import std::time\n\
        let now := time.now_ms()\n\
        now > 0\n";
    let value = run_package(temp.path(), "main.enk", source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_log_emit() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let source = "import std::log\n\
        log.info(\"hello\")\n\
        log.warn(\"warn\")\n\
        log.error(\"err\")\n\
        true\n";
    let value = run_package(temp.path(), "main.enk", source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_io_read_write_text() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let data_path = temp.path().join("data.txt");
    let data_path = data_path.to_string_lossy().replace('\\', "/");
    let source = format!(
        "import std::io\n\
        native::import \"enkai_native\" ::\n    fn buffer_from_string(data: String) -> Buffer\n    fn buffer_eq(a: Buffer, b: Buffer) -> Bool\n::\n\
        io.write_text(\"{}\", \"hello\")\n\
        let text := io.read_text(\"{}\")?\n\
        let expected := buffer_from_string(\"hello\")\n\
        let actual := buffer_from_string(text)\n\
        buffer_eq(expected, actual)\n",
        data_path, data_path
    );
    let value = run_package(temp.path(), "main.enk", &source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_process_run_echo() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let (cmd, args) = if cfg!(windows) {
        ("cmd", "[\"/C\",\"echo\",\"hello\"]")
    } else {
        ("sh", "[\"-c\",\"echo hello\"]")
    };
    let source = format!(
        "import std::process\n\
        native::import \"enkai_native\" ::\n    fn buffer_from_string(data: String) -> Buffer\n    fn buffer_eq(a: Buffer, b: Buffer) -> Bool\n::\n\
        let out := process.run(\"{}\", {}, none)\n\
        out.stdout\n",
        cmd, args
    );
    let value = run_package(temp.path(), "main.enk", &source).expect("run");
    let stdout = match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::String(text) => text.clone(),
            _ => panic!("expected stdout string"),
        },
        _ => panic!("expected stdout string"),
    };
    let normalized = stdout.replace("\r\n", "\n");
    assert_eq!(normalized, "hello\n");
}

#[test]
fn policy_blocks_fs_without_allow() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let data_path = temp.path().join("data.txt");
    fs::write(&data_path, "hello").expect("write data");
    let data_path = data_path.to_string_lossy().replace('\\', "/");
    let source = format!(
        "import std::io\n\
        io.read_text(\"{}\")\n",
        data_path
    );
    let result = run_package_raw(temp.path(), "main.enk", &source);
    assert!(result.is_err());
    let message = result.err().unwrap().to_string();
    assert!(message.contains("[E_POLICY_DENIED]"));
    assert!(message.contains("Policy denied"));
}

#[test]
fn policy_blocks_process_without_allow() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let (cmd, args) = if cfg!(windows) {
        ("cmd", "[\"/C\",\"echo\",\"hello\"]")
    } else {
        ("sh", "[\"-c\",\"echo hello\"]")
    };
    let source = format!(
        "import std::process\n\
        process.run(\"{}\", {}, none)\n",
        cmd, args
    );
    let result = run_package_raw(temp.path(), "main.enk", &source);
    assert!(result.is_err());
    let message = result.err().unwrap().to_string();
    assert!(message.contains("[E_POLICY_DENIED]"));
}

#[test]
fn policy_blocks_db_without_allow() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let db_path = temp.path().join("state.db");
    let db_path = db_path.to_string_lossy().replace('\\', "/");
    let source = format!(
        "import std::db\n\
        db.sqlite_open(\"{}\")\n",
        db_path
    );
    let result = run_package_raw(temp.path(), "main.enk", &source);
    assert!(result.is_err());
    let message = result.err().unwrap().to_string();
    assert!(message.contains("[E_POLICY_DENIED]"));
}

#[test]
fn std_db_sqlite_roundtrip() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let db_path = temp.path().join("state.db");
    let db_path = db_path.to_string_lossy().replace('\\', "/");
    let source = format!(
        "import std::db\n\
        let h := db.sqlite_open(\"{}\")?\n\
        db.sqlite_exec(h, \"create table if not exists items(id integer primary key, name text)\")\n\
        db.sqlite_exec(h, \"delete from items\")\n\
        db.sqlite_exec(h, \"insert into items(name) values ('hello')\")\n\
        let rows := db.sqlite_query(h, \"select name from items order by id\")\n\
        db.sqlite_close(h)\n\
        rows\n",
        db_path
    );
    let value = run_package(temp.path(), "main.enk", &source).expect("run");
    let rows = match value {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::List(items) => items.borrow().clone(),
            _ => panic!("expected rows list"),
        },
        _ => panic!("expected rows list"),
    };
    assert!(!rows.is_empty());
    let first = match &rows[0] {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::Record(map) => map.borrow().get("name").cloned().expect("name"),
            _ => panic!("expected row record"),
        },
        _ => panic!("expected row record"),
    };
    let first_name = match first {
        Value::Obj(obj) => match obj.as_obj() {
            Obj::String(text) => text.clone(),
            _ => panic!("expected name string"),
        },
        _ => panic!("expected name string"),
    };
    assert_eq!(first_name, "hello");
}

#[test]
fn std_db_postgres_open_failure_is_none() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let source = "import std::db\n\
        let h := db.pg_open(\"host=127.0.0.1 port=1 user=enkai password=enkai dbname=enkai connect_timeout=1\")\n\
        h == none\n";
    let value = run_package(temp.path(), "main.enk", source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_analysis_csv_group_and_describe() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let csv_path = temp.path().join("rows.csv");
    fs::write(&csv_path, "team,score\nred,10\nblue,5\nred,3\nblue,7\n").expect("write csv");
    let csv_path = csv_path.to_string_lossy().replace('\\', "/");
    let source = format!(
        "import std::analysis\n\
        let rows := analysis.read_csv(\"{}\", \",\", true)\n\
        let grouped := analysis.group_sum(rows, \"team\", \"score\")\n\
        let stats := analysis.describe([10, 5, 3, 7])\n\
        let nonempty := grouped != []\n\
        nonempty and stats.count == 4 and stats.max > 9\n",
        csv_path
    );
    let value = run_package(temp.path(), "main.enk", &source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_analysis_schema_join_and_pipeline() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let source = "import std::analysis\n\
        let rows := json.parse(\"[{\\\"team\\\":\\\"red\\\",\\\"score\\\":\\\"10\\\"},{\\\"team\\\":\\\"blue\\\",\\\"score\\\":\\\"5\\\"},{\\\"team\\\":\\\"red\\\",\\\"score\\\":\\\"3\\\"}]\")\n\
        let schema := json.parse(\"{\\\"team\\\":\\\"string\\\",\\\"score\\\":{\\\"type\\\":\\\"int\\\"}}\")\n\
        let checked := analysis.validate_schema(rows, schema)\n\
        let inferred := analysis.infer_schema_typed(checked.rows)\n\
        let right := json.parse(\"[{\\\"team\\\":\\\"red\\\",\\\"region\\\":\\\"west\\\"},{\\\"team\\\":\\\"blue\\\",\\\"region\\\":\\\"east\\\"}]\")\n\
        let joined := analysis.join(checked.rows, right, \"team\", \"team\", \"inner\")\n\
        let quant := analysis.quantiles([1,2,3,4,5], [0.5, 0.9])\n\
        let rolling := analysis.rolling_mean([1,2,3,4], 2)\n\
        let pipeline := json.parse(\"{\\\"name\\\":\\\"sum_by_team\\\",\\\"stages\\\":[{\\\"op\\\":\\\"group_agg\\\",\\\"key\\\":\\\"team\\\",\\\"field\\\":\\\"score\\\",\\\"agg\\\":\\\"sum\\\"}]}\")\n\
        let run := analysis.run_pipeline(checked.rows, pipeline)\n\
        checked.ok and inferred.schema_version == 1 and joined != [] and quant != [] and rolling != [] and run.manifest.schema_version == 1 and run.manifest.stage_count == 1\n";
    let value = run_package(temp.path(), "main.enk", source).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn std_algo_sort_shortest_path_and_metrics() {
    let temp = tempfile::tempdir().expect("tempdir");
    copy_std_modules(temp.path());
    let source = "import std::algo\n\
        let sorted := algo.sort_ints([4, 1, 3, 2])\n\
        let idx := algo.binary_search_ints(sorted, 3)\n\
        let edges := json.parse(\"[{\\\"from\\\":\\\"a\\\",\\\"to\\\":\\\"b\\\",\\\"weight\\\":1},{\\\"from\\\":\\\"b\\\",\\\"to\\\":\\\"c\\\",\\\"weight\\\":2},{\\\"from\\\":\\\"a\\\",\\\"to\\\":\\\"c\\\",\\\"weight\\\":10}]\")\n\
        let path := algo.shortest_path(edges, \"a\", \"c\")\n\
        let acc := algo.accuracy([1, 0, 1], [1, 1, 1])\n\
        idx == 2 and path.reachable and path.distance > 2 and path.distance < 4 and acc > 0.6\n";
    let value = run_package(temp.path(), "main.enk", source).expect("run");
    assert_eq!(value, Value::Bool(true));
}
