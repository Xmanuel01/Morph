use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use enkai_compiler::bytecode::Program;
use enkai_compiler::compiler::{compile_module, compile_package, CompileError};
use enkai_compiler::modules::load_package;
use enkai_compiler::parse_module_named;
use enkai_compiler::{TypeChecker, TypeError};
use enkai_runtime::object::Obj;
use enkai_runtime::{Value, VM};

const FMT_LITE_SCRIPT: &str = include_str!("../tools/bootstrap/fmt_lite.enk");
const LINT_LITE_SCRIPT: &str = include_str!("../tools/bootstrap/lint_lite.enk");
const TOKENIZER_LITE_SCRIPT: &str = include_str!("../tools/bootstrap/tokenizer_lite.enk");
const DATASET_LITE_SCRIPT: &str = include_str!("../tools/bootstrap/dataset_lite.enk");
const ENKAI_LITEC_SCRIPT: &str = include_str!("../tools/bootstrap/enkai_lite.enk");

pub fn print_usage() {
    eprintln!("  enkai fmt-lite [--check] <file|dir>");
    eprintln!("  enkai lint-lite [--deny-warn] <file|dir>");
    eprintln!("  enkai tokenizer-lite train <dataset_path> <tokenizer_path> [--vocab-size <n>] [--min-freq <n>] [--seed <n>] [--lowercase]");
    eprintln!("  enkai dataset-lite inspect <dataset_path> <tokenizer_path> --seq-len <n> --batch-size <n> [--seed <n>] [--shuffle] [--drop-remainder|--keep-remainder] [--no-add-eos] [--prefetch-batches <n>] [--output <path>]");
    eprintln!("  enkai litec check <input.enk>");
    eprintln!("  enkai litec compile <input.enk> --out <program.bin>");
    eprintln!("  enkai litec verify <input.enk>");
    eprintln!("  enkai litec stage <parse|check|codegen> <input.enk> [--out <program.bin>]");
    eprintln!("  enkai litec selfhost <corpus_dir>");
    eprintln!("  enkai litec selfhost-ci <corpus_dir> [--no-compare-stage0]");
}

pub fn fmt_lite_command(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("enkai fmt-lite requires a file or directory");
        return 1;
    }
    let mut check = false;
    let mut target: Option<PathBuf> = None;
    for arg in args {
        if arg == "--check" {
            check = true;
        } else if target.is_none() {
            target = Some(PathBuf::from(arg));
        } else {
            eprintln!("Unexpected argument: {}", arg);
            return 1;
        }
    }
    let target = match target {
        Some(path) => path,
        None => {
            eprintln!("enkai fmt-lite requires a file or directory");
            return 1;
        }
    };
    let files = match super::collect_source_files(&target) {
        Ok(files) => files,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let mut failed = false;
    for file in files {
        let code = match run_embedded_script(
            "fmt_lite.enk",
            FMT_LITE_SCRIPT,
            &[
                ("ENKAI_BOOTSTRAP_FILE", file.to_string_lossy().to_string()),
                (
                    "ENKAI_FMT_LITE_CHECK",
                    if check { "1" } else { "0" }.to_string(),
                ),
            ],
        ) {
            Ok(code) => code,
            Err(err) => {
                eprintln!("fmt-lite failed: {}", err);
                return 1;
            }
        };
        if code != 0 {
            failed = true;
        }
    }
    if failed {
        1
    } else {
        0
    }
}

pub fn lint_lite_command(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("enkai lint-lite requires a file or directory");
        return 1;
    }
    let mut deny_warn = false;
    let mut target: Option<PathBuf> = None;
    for arg in args {
        if arg == "--deny-warn" {
            deny_warn = true;
        } else if target.is_none() {
            target = Some(PathBuf::from(arg));
        } else {
            eprintln!("Unexpected argument: {}", arg);
            return 1;
        }
    }
    let target = match target {
        Some(path) => path,
        None => {
            eprintln!("enkai lint-lite requires a file or directory");
            return 1;
        }
    };
    let files = match super::collect_source_files(&target) {
        Ok(files) => files,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let mut has_warnings = false;
    for file in files {
        let code = match run_embedded_script(
            "lint_lite.enk",
            LINT_LITE_SCRIPT,
            &[("ENKAI_BOOTSTRAP_FILE", file.to_string_lossy().to_string())],
        ) {
            Ok(code) => code,
            Err(err) => {
                eprintln!("lint-lite failed: {}", err);
                return 1;
            }
        };
        if code == 1 {
            has_warnings = true;
            if deny_warn {
                return 1;
            }
        } else if code != 0 {
            return 1;
        }
    }
    if deny_warn && has_warnings {
        1
    } else {
        0
    }
}

pub fn tokenizer_lite_command(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("enkai tokenizer-lite requires a subcommand");
        return 1;
    }
    match args[0].as_str() {
        "train" => tokenizer_lite_train(&args[1..]),
        other => {
            eprintln!("unknown tokenizer-lite subcommand: {}", other);
            1
        }
    }
}

fn tokenizer_lite_train(args: &[String]) -> i32 {
    if args.len() < 2 {
        eprintln!(
            "Usage: enkai tokenizer-lite train <dataset_path> <tokenizer_path> [--vocab-size <n>] [--min-freq <n>] [--seed <n>] [--lowercase]"
        );
        return 1;
    }
    let dataset_path = args[0].clone();
    let tokenizer_path = args[1].clone();
    let mut vocab_size: Option<i64> = None;
    let mut min_freq: Option<i64> = None;
    let mut seed: Option<i64> = None;
    let mut lowercase = false;
    let mut index = 2usize;
    while index < args.len() {
        match args[index].as_str() {
            "--vocab-size" => {
                index += 1;
                let value = match args.get(index) {
                    Some(value) => value,
                    None => {
                        eprintln!("--vocab-size requires a value");
                        return 1;
                    }
                };
                let parsed = match value.parse::<i64>() {
                    Ok(value) if value > 0 => value,
                    _ => {
                        eprintln!("--vocab-size must be a positive integer");
                        return 1;
                    }
                };
                vocab_size = Some(parsed);
            }
            "--min-freq" => {
                index += 1;
                let value = match args.get(index) {
                    Some(value) => value,
                    None => {
                        eprintln!("--min-freq requires a value");
                        return 1;
                    }
                };
                let parsed = match value.parse::<i64>() {
                    Ok(value) if value > 0 => value,
                    _ => {
                        eprintln!("--min-freq must be a positive integer");
                        return 1;
                    }
                };
                min_freq = Some(parsed);
            }
            "--seed" => {
                index += 1;
                let value = match args.get(index) {
                    Some(value) => value,
                    None => {
                        eprintln!("--seed requires a value");
                        return 1;
                    }
                };
                let parsed = match value.parse::<i64>() {
                    Ok(value) if value >= 0 => value,
                    _ => {
                        eprintln!("--seed must be a non-negative integer");
                        return 1;
                    }
                };
                seed = Some(parsed);
            }
            "--lowercase" => {
                lowercase = true;
            }
            other => {
                eprintln!("Unexpected argument: {}", other);
                return 1;
            }
        }
        index += 1;
    }
    let mut envs = vec![
        ("ENKAI_TOKENIZER_LITE_DATASET", dataset_path),
        ("ENKAI_TOKENIZER_LITE_OUTPUT", tokenizer_path),
        (
            "ENKAI_TOKENIZER_LITE_LOWERCASE",
            if lowercase { "1" } else { "0" }.to_string(),
        ),
    ];
    if let Some(value) = vocab_size {
        envs.push(("ENKAI_TOKENIZER_LITE_VOCAB_SIZE", value.to_string()));
    }
    if let Some(value) = min_freq {
        envs.push(("ENKAI_TOKENIZER_LITE_MIN_FREQ", value.to_string()));
    }
    if let Some(value) = seed {
        envs.push(("ENKAI_TOKENIZER_LITE_SEED", value.to_string()));
    }
    match run_embedded_script("tokenizer_lite.enk", TOKENIZER_LITE_SCRIPT, &envs) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("tokenizer-lite failed: {}", err);
            1
        }
    }
}

pub fn dataset_lite_command(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("enkai dataset-lite requires a subcommand");
        return 1;
    }
    match args[0].as_str() {
        "inspect" => dataset_lite_inspect(&args[1..]),
        other => {
            eprintln!("unknown dataset-lite subcommand: {}", other);
            1
        }
    }
}

fn dataset_lite_inspect(args: &[String]) -> i32 {
    if args.len() < 2 {
        eprintln!(
            "Usage: enkai dataset-lite inspect <dataset_path> <tokenizer_path> --seq-len <n> --batch-size <n> [--seed <n>] [--shuffle] [--drop-remainder|--keep-remainder] [--no-add-eos] [--prefetch-batches <n>] [--output <path>]"
        );
        return 1;
    }
    let dataset_path = args[0].clone();
    let tokenizer_path = args[1].clone();
    let mut seq_len: Option<i64> = None;
    let mut batch_size: Option<i64> = None;
    let mut seed: Option<i64> = None;
    let mut shuffle = false;
    let mut add_eos = true;
    let mut drop_remainder = true;
    let mut prefetch_batches: i64 = 0;
    let mut output_path: Option<String> = None;
    let mut index = 2usize;
    while index < args.len() {
        match args[index].as_str() {
            "--seq-len" => {
                index += 1;
                let value = match args.get(index) {
                    Some(value) => value,
                    None => {
                        eprintln!("--seq-len requires a value");
                        return 1;
                    }
                };
                let parsed = match value.parse::<i64>() {
                    Ok(value) if value > 0 => value,
                    _ => {
                        eprintln!("--seq-len must be a positive integer");
                        return 1;
                    }
                };
                seq_len = Some(parsed);
            }
            "--batch-size" => {
                index += 1;
                let value = match args.get(index) {
                    Some(value) => value,
                    None => {
                        eprintln!("--batch-size requires a value");
                        return 1;
                    }
                };
                let parsed = match value.parse::<i64>() {
                    Ok(value) if value > 0 => value,
                    _ => {
                        eprintln!("--batch-size must be a positive integer");
                        return 1;
                    }
                };
                batch_size = Some(parsed);
            }
            "--seed" => {
                index += 1;
                let value = match args.get(index) {
                    Some(value) => value,
                    None => {
                        eprintln!("--seed requires a value");
                        return 1;
                    }
                };
                let parsed = match value.parse::<i64>() {
                    Ok(value) if value >= 0 => value,
                    _ => {
                        eprintln!("--seed must be a non-negative integer");
                        return 1;
                    }
                };
                seed = Some(parsed);
            }
            "--shuffle" => {
                shuffle = true;
            }
            "--drop-remainder" => {
                drop_remainder = true;
            }
            "--keep-remainder" => {
                drop_remainder = false;
            }
            "--no-add-eos" => {
                add_eos = false;
            }
            "--prefetch-batches" => {
                index += 1;
                let value = match args.get(index) {
                    Some(value) => value,
                    None => {
                        eprintln!("--prefetch-batches requires a value");
                        return 1;
                    }
                };
                let parsed = match value.parse::<i64>() {
                    Ok(value) if value >= 0 => value,
                    _ => {
                        eprintln!("--prefetch-batches must be a non-negative integer");
                        return 1;
                    }
                };
                prefetch_batches = parsed;
            }
            "--output" => {
                index += 1;
                let value = match args.get(index) {
                    Some(value) => value,
                    None => {
                        eprintln!("--output requires a value");
                        return 1;
                    }
                };
                output_path = Some(value.clone());
            }
            other => {
                eprintln!("Unexpected argument: {}", other);
                return 1;
            }
        }
        index += 1;
    }
    let seq_len = match seq_len {
        Some(value) => value,
        None => {
            eprintln!("--seq-len is required");
            return 1;
        }
    };
    let batch_size = match batch_size {
        Some(value) => value,
        None => {
            eprintln!("--batch-size is required");
            return 1;
        }
    };
    let mut envs = vec![
        ("ENKAI_DATASET_LITE_DATASET", dataset_path),
        ("ENKAI_DATASET_LITE_TOKENIZER", tokenizer_path),
        ("ENKAI_DATASET_LITE_SEQ_LEN", seq_len.to_string()),
        ("ENKAI_DATASET_LITE_BATCH_SIZE", batch_size.to_string()),
        (
            "ENKAI_DATASET_LITE_SHUFFLE",
            if shuffle { "1" } else { "0" }.to_string(),
        ),
        (
            "ENKAI_DATASET_LITE_ADD_EOS",
            if add_eos { "1" } else { "0" }.to_string(),
        ),
        (
            "ENKAI_DATASET_LITE_DROP_REMAINDER",
            if drop_remainder { "1" } else { "0" }.to_string(),
        ),
        (
            "ENKAI_DATASET_LITE_PREFETCH_BATCHES",
            prefetch_batches.to_string(),
        ),
    ];
    if let Some(value) = seed {
        envs.push(("ENKAI_DATASET_LITE_SEED", value.to_string()));
    }
    if let Some(value) = output_path {
        envs.push(("ENKAI_DATASET_LITE_OUTPUT", value));
    }
    match run_embedded_script("dataset_lite.enk", DATASET_LITE_SCRIPT, &envs) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("dataset-lite failed: {}", err);
            1
        }
    }
}

pub fn litec_command(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("enkai litec requires a subcommand");
        return 1;
    }
    match args[0].as_str() {
        "check" => litec_check(&args[1..]),
        "compile" => litec_compile(&args[1..]),
        "verify" => litec_verify(&args[1..]),
        "stage" => litec_stage(&args[1..]),
        "selfhost" => litec_selfhost(&args[1..]),
        "selfhost-ci" => litec_selfhost_ci(&args[1..]),
        other => {
            eprintln!("unknown litec subcommand: {}", other);
            1
        }
    }
}

fn litec_check(args: &[String]) -> i32 {
    if args.len() != 1 {
        eprintln!("Usage: enkai litec check <input.enk>");
        return 1;
    }
    let input = PathBuf::from(&args[0]);
    if !input.is_file() {
        eprintln!("input file not found: {}", input.display());
        return 1;
    }
    match run_litec_mode("check", &input, None) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("litec check failed: {}", err);
            1
        }
    }
}

fn litec_compile(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("Usage: enkai litec compile <input.enk> --out <program.bin>");
        return 1;
    }
    let input = PathBuf::from(&args[0]);
    if !input.is_file() {
        eprintln!("input file not found: {}", input.display());
        return 1;
    }
    let mut output: Option<PathBuf> = None;
    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--out" => {
                index += 1;
                let Some(path) = args.get(index) else {
                    eprintln!("--out requires a value");
                    return 1;
                };
                output = Some(PathBuf::from(path));
            }
            other => {
                eprintln!("Unexpected argument: {}", other);
                return 1;
            }
        }
        index += 1;
    }
    let output = match output {
        Some(path) => path,
        None => {
            eprintln!("Usage: enkai litec compile <input.enk> --out <program.bin>");
            return 1;
        }
    };
    match run_litec_mode("compile", &input, Some(&output)) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("litec compile failed: {}", err);
            1
        }
    }
}

fn litec_verify(args: &[String]) -> i32 {
    if args.len() != 1 {
        eprintln!("Usage: enkai litec verify <input.enk>");
        return 1;
    }
    let input = PathBuf::from(&args[0]);
    if !input.is_file() {
        eprintln!("input file not found: {}", input.display());
        return 1;
    }
    match verify_stage_equivalence(&input) {
        Ok(()) => {
            println!("litec verify ok");
            0
        }
        Err(err) => {
            eprintln!("{}", err);
            1
        }
    }
}

fn litec_stage(args: &[String]) -> i32 {
    if args.len() < 2 {
        eprintln!(
            "Usage: enkai litec stage <parse|check|codegen> <input.enk> [--out <program.bin>]"
        );
        return 1;
    }
    let mode = match args[0].as_str() {
        "parse" => "parse",
        "check" => "check",
        "codegen" => "codegen",
        other => {
            eprintln!(
                "unknown litec stage '{}', expected parse|check|codegen",
                other
            );
            return 1;
        }
    };
    let input = PathBuf::from(&args[1]);
    if !input.is_file() {
        eprintln!("input file not found: {}", input.display());
        return 1;
    }
    let mut output: Option<PathBuf> = None;
    let mut index = 2usize;
    while index < args.len() {
        match args[index].as_str() {
            "--out" => {
                index += 1;
                let Some(path) = args.get(index) else {
                    eprintln!("--out requires a value");
                    return 1;
                };
                output = Some(PathBuf::from(path));
            }
            other => {
                eprintln!("Unexpected argument: {}", other);
                return 1;
            }
        }
        index += 1;
    }
    if mode == "codegen" && output.is_none() {
        eprintln!("enkai litec stage codegen requires --out <program.bin>");
        return 1;
    }
    if mode != "codegen" && output.is_some() {
        eprintln!("--out is only valid for codegen stage");
        return 1;
    }
    match run_litec_mode(mode, &input, output.as_deref()) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("litec stage failed: {}", err);
            1
        }
    }
}

fn litec_selfhost(args: &[String]) -> i32 {
    if args.len() != 1 {
        eprintln!("Usage: enkai litec selfhost <corpus_dir>");
        return 1;
    }
    let corpus = PathBuf::from(&args[0]);
    if !corpus.is_dir() {
        eprintln!("corpus directory not found: {}", corpus.display());
        return 1;
    }
    let files = match super::collect_source_files(&corpus) {
        Ok(files) => files,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    if files.is_empty() {
        eprintln!("no source files found in {}", corpus.display());
        return 1;
    }
    for file in &files {
        if let Err(err) = verify_stage_equivalence(file) {
            eprintln!("selfhost mismatch for {}: {}", file.display(), err);
            return 1;
        }
    }
    println!("litec selfhost ok ({} files)", files.len());
    0
}

fn litec_selfhost_ci(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("Usage: enkai litec selfhost-ci <corpus_dir> [--no-compare-stage0]");
        return 1;
    }
    let corpus = PathBuf::from(&args[0]);
    if !corpus.is_dir() {
        eprintln!("corpus directory not found: {}", corpus.display());
        return 1;
    }
    let mut compare_stage0 = true;
    for arg in &args[1..] {
        match arg.as_str() {
            "--no-compare-stage0" => compare_stage0 = false,
            other => {
                eprintln!("Unexpected argument: {}", other);
                return 1;
            }
        }
    }
    let files = match super::collect_source_files(&corpus) {
        Ok(files) => files,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    if files.is_empty() {
        eprintln!("no source files found in {}", corpus.display());
        return 1;
    }
    let mut passed = 0usize;
    for file in &files {
        let stage1_bytes = match compile_stage1_program_bytes(file, "codegen") {
            Ok(bytes) => bytes,
            Err(err) => {
                eprintln!("[fail] {}: {}", file.display(), err);
                return 1;
            }
        };
        let stage1 = match decode_program_bytes(file, &stage1_bytes, "stage1") {
            Ok(program) => program,
            Err(err) => {
                eprintln!("[fail] {}: {}", file.display(), err);
                return 1;
            }
        };
        let stage1_value = match run_program(&stage1) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("[fail] {}: stage1 runtime error: {}", file.display(), err);
                return 1;
            }
        };
        if compare_stage0 {
            let stage0_bytes = match compile_stage0_program_bytes(file) {
                Ok(bytes) => bytes,
                Err(err) => {
                    eprintln!("[fail] {}: stage0 compile failed: {}", file.display(), err);
                    return 1;
                }
            };
            let stage0 = match decode_program_bytes(file, &stage0_bytes, "stage0") {
                Ok(program) => program,
                Err(err) => {
                    eprintln!("[fail] {}: {}", file.display(), err);
                    return 1;
                }
            };
            let stage0_value = match run_program(&stage0) {
                Ok(value) => value,
                Err(err) => {
                    eprintln!("[fail] {}: stage0 runtime error: {}", file.display(), err);
                    return 1;
                }
            };
            if canonical_value(&stage0_value) != canonical_value(&stage1_value) {
                eprintln!(
                    "[fail] {}: stage0/stage1 result mismatch (stage0={}, stage1={})",
                    file.display(),
                    canonical_value(&stage0_value),
                    canonical_value(&stage1_value)
                );
                return 1;
            }
        }
        println!("[pass] {}", file.display());
        passed += 1;
    }
    println!(
        "litec selfhost-ci ok ({} files, compare_stage0={})",
        passed, compare_stage0
    );
    0
}

fn verify_stage_equivalence(input: &Path) -> Result<(), String> {
    let stage0 = match compile_stage0_program_bytes(input) {
        Ok(bytes) => bytes,
        Err(err) => {
            return Err(format!("stage0 compile failed: {}", err));
        }
    };
    let stage1 = compile_stage1_program_bytes(input, "compile")?;
    if stage0 != stage1 {
        return Err(format!(
            "stage0/stage1 bytecode mismatch (stage0={} bytes, stage1={} bytes)",
            stage0.len(),
            stage1.len()
        ));
    }
    Ok(())
}

fn compile_stage1_program_bytes(input: &Path, mode: &str) -> Result<Vec<u8>, String> {
    let stage1_path = temp_stage1_program_path()
        .map_err(|err| format!("failed to allocate stage1 path: {}", err))?;
    let code = match run_litec_mode(mode, input, Some(&stage1_path)) {
        Ok(code) => code,
        Err(err) => {
            let _ = fs::remove_file(&stage1_path);
            return Err(format!("stage1 compile failed: {}", err));
        }
    };
    if code != 0 {
        let _ = fs::remove_file(&stage1_path);
        return Err(format!("stage1 compile returned non-zero status {}", code));
    }
    let stage1 = fs::read(&stage1_path).map_err(|err| {
        let _ = fs::remove_file(&stage1_path);
        format!(
            "failed to read stage1 output {}: {}",
            stage1_path.display(),
            err
        )
    })?;
    let _ = fs::remove_file(&stage1_path);
    Ok(stage1)
}

fn decode_program_bytes(input: &Path, bytes: &[u8], label: &str) -> Result<Program, String> {
    bincode::deserialize::<Program>(bytes)
        .map_err(|err| format!("{} decode failed for {}: {}", label, input.display(), err))
}

fn run_program(program: &Program) -> Result<Value, String> {
    let mut vm = VM::new(false, false, false, false);
    vm.run(program).map_err(|err| err.to_string())
}

fn canonical_value(value: &Value) -> String {
    match value {
        Value::Int(v) => format!("Int({})", v),
        Value::Float(v) => format!("Float({})", v),
        Value::Bool(v) => format!("Bool({})", v),
        Value::Null => "Null".to_string(),
        Value::Obj(obj) => match obj.as_obj() {
            Obj::String(text) => format!("String({:?})", text),
            Obj::Buffer(bytes) => format!("Buffer({:?})", bytes),
            Obj::List(items) => {
                let values = items
                    .borrow()
                    .iter()
                    .map(canonical_value)
                    .collect::<Vec<_>>();
                format!("List([{}])", values.join(","))
            }
            Obj::Record(map) => {
                let map = map.borrow();
                let mut keys = map.keys().cloned().collect::<Vec<_>>();
                keys.sort();
                let mut pairs = Vec::with_capacity(keys.len());
                for key in keys {
                    if let Some(value) = map.get(&key) {
                        pairs.push(format!("{}={}", key, canonical_value(value)));
                    }
                }
                format!("Record({{{}}})", pairs.join(","))
            }
            Obj::Json(value) => format!("Json({})", value),
            Obj::Function(func) => format!("Function({:?}/{})", func.name, func.arity),
            Obj::BoundFunction(func) => {
                format!(
                    "BoundFunction(func={},arity={})",
                    func.func_index, func.arity
                )
            }
            Obj::NativeFunction(func) => format!("NativeFunction({}/{})", func.name, func.arity),
            Obj::TaskHandle(id) => format!("TaskHandle({})", id),
            Obj::Channel(_) => "Channel".to_string(),
            Obj::TcpListener(_) => "TcpListener".to_string(),
            Obj::TcpConnection(_) => "TcpConnection".to_string(),
            Obj::HttpStream(_) => "HttpStream".to_string(),
            Obj::Tokenizer(_) => "Tokenizer".to_string(),
            Obj::DatasetStream(_) => "DatasetStream".to_string(),
        },
    }
}

fn temp_stage1_program_path() -> Result<PathBuf, String> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| err.to_string())?
        .as_nanos();
    let mut path = env::temp_dir();
    path.push(format!(
        "enkai_litec_stage1_{}_{}.bin",
        std::process::id(),
        nonce
    ));
    Ok(path)
}

fn run_litec_mode(mode: &str, input: &Path, output: Option<&Path>) -> Result<i32, String> {
    let mut envs = vec![
        ("ENKAI_LITEC_MODE", mode.to_string()),
        (
            "ENKAI_LITEC_INPUT",
            input
                .canonicalize()
                .unwrap_or_else(|_| input.to_path_buf())
                .to_string_lossy()
                .to_string(),
        ),
    ];
    if let Some(output) = output {
        envs.push((
            "ENKAI_LITEC_OUTPUT",
            output
                .canonicalize()
                .unwrap_or_else(|_| output.to_path_buf())
                .to_string_lossy()
                .to_string(),
        ));
    }
    run_embedded_script("enkai_lite.enk", ENKAI_LITEC_SCRIPT, &envs)
}

fn compile_stage0_program_bytes(input: &Path) -> Result<Vec<u8>, String> {
    let source = fs::read_to_string(input)
        .map_err(|err| format!("Failed to read {}: {}", input.display(), err))?;
    let source_name = input.to_string_lossy();
    let module = parse_module_named(&source, Some(source_name.as_ref()))
        .map_err(|err| format!("Parse error: {}", err))?;
    let mut checker = TypeChecker::new();
    checker
        .check_module(&module)
        .map_err(|err| type_error_to_string(&err))?;
    let program = compile_module(&module).map_err(|err| compile_error_to_string(&err))?;
    bincode::serialize(&program).map_err(|err| format!("serialize failed: {}", err))
}

fn run_embedded_script(
    name: &str,
    source: &str,
    overrides: &[(&str, String)],
) -> Result<i32, String> {
    let lock = super::env_guard();
    let script = write_temp_script(name, source)?;
    let script_path = script.path.clone();
    let mut env_guards: Vec<EnvOverride> = Vec::new();
    for (key, value) in overrides {
        env_guards.push(EnvOverride::set(key, value));
    }
    if env::var("ENKAI_STD").is_err() {
        if let Some(std_path) = detect_std_path() {
            env_guards.push(EnvOverride::set(
                "ENKAI_STD",
                std_path.to_string_lossy().as_ref(),
            ));
        }
    }
    let package = load_package(&script_path).map_err(|err| err.to_string())?;
    TypeChecker::check_package(&package).map_err(|err| type_error_to_string(&err))?;
    let program = compile_package(&package).map_err(|err| compile_error_to_string(&err))?;
    let mut vm = VM::new(false, false, false, false);
    let value = vm.run(&program).map_err(|err| err.to_string())?;
    drop(env_guards);
    drop(lock);
    drop(script);
    match value {
        Value::Int(code) => Ok(code as i32),
        Value::Null => Ok(0),
        _ => Err("bootstrap script returned non-int result".to_string()),
    }
}

fn write_temp_script(name: &str, source: &str) -> Result<TempScript, String> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| err.to_string())?
        .as_nanos();
    let mut path = env::temp_dir();
    path.push(format!(
        "enkai_bootstrap_{}_{}_{}",
        std::process::id(),
        nonce,
        name
    ));
    fs::write(&path, source)
        .map_err(|err| format!("Failed to write {}: {}", path.display(), err))?;
    Ok(TempScript { path })
}

fn detect_std_path() -> Option<PathBuf> {
    let workspace_std = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|root| root.join("std"))?;
    if workspace_std.is_dir() {
        return Some(workspace_std);
    }
    let exe = env::current_exe().ok()?;
    if let Some(dir) = exe.parent() {
        let direct = dir.join("std");
        if direct.is_dir() {
            return Some(direct);
        }
        if let Some(parent) = dir.parent() {
            let sibling = parent.join("std");
            if sibling.is_dir() {
                return Some(sibling);
            }
        }
    }
    None
}

fn type_error_to_string(err: &TypeError) -> String {
    if let Some(diagnostic) = err.diagnostic() {
        diagnostic.to_string()
    } else {
        format!(
            "Type error: {} at {}:{}",
            err.message, err.span.line, err.span.col
        )
    }
}

fn compile_error_to_string(err: &CompileError) -> String {
    if let Some(diagnostic) = err.diagnostic() {
        diagnostic.to_string()
    } else if let Some(span) = &err.span {
        format!(
            "Compile error: {} at {}:{}",
            err.message, span.line, span.col
        )
    } else {
        format!("Compile error: {}", err.message)
    }
}

struct EnvOverride {
    key: String,
    prev: Option<String>,
}

struct TempScript {
    path: PathBuf,
}

impl Drop for TempScript {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

impl EnvOverride {
    fn set(key: &str, value: &str) -> Self {
        let prev = env::var(key).ok();
        env::set_var(key, value);
        Self {
            key: key.to_string(),
            prev,
        }
    }
}

impl Drop for EnvOverride {
    fn drop(&mut self) {
        if let Some(value) = &self.prev {
            env::set_var(&self.key, value);
        } else {
            env::remove_var(&self.key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;

    use enkai_runtime::dataset::{resolve_dataset_paths, DatasetConfig, DatasetStream};
    use enkai_runtime::tokenizer::{Tokenizer, TrainConfig};
    use tempfile::tempdir;

    #[test]
    fn fmt_lite_formats_like_rust_formatter() {
        let dir = tempdir().expect("tempdir");
        let file_lite = dir.path().join("lite.enk");
        let file_rust = dir.path().join("rust.enk");
        let source = "if true ::\nprint(\"hi\")\n::\n";
        fs::write(&file_lite, source).expect("write lite");
        fs::write(&file_rust, source).expect("write rust");
        let lite_code = fmt_lite_command(&[file_lite.to_string_lossy().to_string()]);
        let rust_code = super::super::fmt_command(&[file_rust.to_string_lossy().to_string()]);
        assert_eq!(lite_code, 0);
        assert_eq!(rust_code, 0);
        let lite_out = fs::read_to_string(file_lite).expect("read lite");
        let rust_out = fs::read_to_string(file_rust).expect("read rust");
        assert_eq!(lite_out, rust_out);
    }

    #[test]
    fn fmt_lite_check_matches_rust_exit_code() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("bad.enk");
        fs::write(&file, "if true ::\nprint(\"hi\")\n::\n").expect("write");
        let lite_code =
            fmt_lite_command(&["--check".to_string(), file.to_string_lossy().to_string()]);
        let rust_code =
            super::super::fmt_command(&["--check".to_string(), file.to_string_lossy().to_string()]);
        assert_eq!(lite_code, rust_code);
    }

    #[test]
    fn lint_lite_deny_warn_enforces_failure() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("lint.enk");
        fs::write(&file, "fn main() ::\n\tprint(\"tab\")   \n::\n").expect("write");
        let soft = lint_lite_command(&[file.to_string_lossy().to_string()]);
        let strict = lint_lite_command(&[
            "--deny-warn".to_string(),
            file.to_string_lossy().to_string(),
        ]);
        assert_eq!(soft, 0);
        assert_eq!(strict, 1);
    }

    #[test]
    fn tokenizer_lite_train_matches_rust_baseline() {
        let dir = tempdir().expect("tempdir");
        let data = dir.path().join("data.txt");
        fs::write(&data, "Hello world\nhello model\n").expect("data");
        let lite_tok = dir.path().join("lite.tokenizer.json");
        let rust_tok = dir.path().join("rust.tokenizer.json");
        let code = tokenizer_lite_command(&[
            "train".to_string(),
            data.to_string_lossy().to_string(),
            lite_tok.to_string_lossy().to_string(),
            "--vocab-size".to_string(),
            "64".to_string(),
            "--min-freq".to_string(),
            "1".to_string(),
            "--seed".to_string(),
            "42".to_string(),
            "--lowercase".to_string(),
        ]);
        assert_eq!(code, 0);
        let cfg = TrainConfig {
            vocab_size: 64,
            min_freq: 1,
            lowercase: true,
            seed: Some(42),
        };
        let rust = Tokenizer::train_from_path(&data, &cfg).expect("rust train");
        rust.save(&rust_tok).expect("rust save");
        let lite = Tokenizer::load(&lite_tok).expect("lite load");
        let rust_loaded = Tokenizer::load(&rust_tok).expect("rust load");
        let probe = "hello world";
        assert_eq!(lite.encode(probe, true), rust_loaded.encode(probe, true));
    }

    #[test]
    fn dataset_lite_inspect_matches_rust_baseline() {
        let dir = tempdir().expect("tempdir");
        let data = dir.path().join("dataset.txt");
        fs::write(
            &data,
            "hello world one\nhello world two\nhello world three\nhello world four\n",
        )
        .expect("dataset");
        let tokenizer_path = dir.path().join("dataset.tokenizer.json");
        let cfg = TrainConfig {
            vocab_size: 128,
            min_freq: 1,
            lowercase: true,
            seed: Some(7),
        };
        let tok = Tokenizer::train_from_path(&data, &cfg).expect("train");
        tok.save(&tokenizer_path).expect("save");
        let summary_path = dir.path().join("summary.json");
        let code = dataset_lite_command(&[
            "inspect".to_string(),
            data.to_string_lossy().to_string(),
            tokenizer_path.to_string_lossy().to_string(),
            "--seq-len".to_string(),
            "6".to_string(),
            "--batch-size".to_string(),
            "2".to_string(),
            "--seed".to_string(),
            "11".to_string(),
            "--output".to_string(),
            summary_path.to_string_lossy().to_string(),
        ]);
        assert_eq!(code, 0);
        let summary_text = fs::read_to_string(&summary_path).expect("summary");
        let summary_json: serde_json::Value =
            serde_json::from_str(&summary_text).expect("summary json");
        let rust_tok = Tokenizer::load(&tokenizer_path).expect("load tok");
        let files = resolve_dataset_paths(data.to_string_lossy().as_ref()).expect("paths");
        let mut dataset_cfg = DatasetConfig::new(6, 2);
        dataset_cfg.seed = Some(11);
        dataset_cfg.add_eos = true;
        dataset_cfg.drop_remainder = true;
        dataset_cfg.shuffle = false;
        dataset_cfg.prefetch_batches = 0;
        let mut stream = DatasetStream::new(files, rust_tok, dataset_cfg).expect("dataset stream");
        let batch = stream
            .next_batch()
            .expect("next batch")
            .expect("batch present");
        assert_eq!(
            summary_json
                .get("batch_size")
                .and_then(|v| v.as_i64())
                .unwrap_or_default(),
            batch.batch_size as i64
        );
        assert_eq!(
            summary_json
                .get("seq_len")
                .and_then(|v| v.as_i64())
                .unwrap_or_default(),
            batch.seq_len as i64
        );
        assert_eq!(
            summary_json
                .get("token_count")
                .and_then(|v| v.as_i64())
                .unwrap_or_default(),
            batch.token_count as i64
        );
        let lite_eff = summary_json
            .get("packing_efficiency")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        assert!((lite_eff - batch.packing_efficiency as f64).abs() < f64::EPSILON);
    }

    #[test]
    fn litec_compile_emits_program() {
        let dir = tempdir().expect("tempdir");
        let input = dir.path().join("input.enk");
        fs::write(
            &input,
            "fn main() -> Int ::\n    let x := 1 + 2\n    return x\n::\nmain()\n",
        )
        .expect("write input");
        let output = dir.path().join("stage1.bin");
        let code = litec_command(&[
            "compile".to_string(),
            input.to_string_lossy().to_string(),
            "--out".to_string(),
            output.to_string_lossy().to_string(),
        ]);
        assert_eq!(code, 0);
        let bytes = fs::read(&output).expect("stage1 output");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn litec_verify_matches_stage0_and_stage1() {
        let dir = tempdir().expect("tempdir");
        let input = dir.path().join("verify.enk");
        fs::write(
            &input,
            "fn main() -> Int ::\n    let n := 3\n    let out := 0\n    while n > 0 ::\n        out := out + n\n        n := n - 1\n    ::\n    return out\n::\nmain()\n",
        )
        .expect("write input");
        let code = litec_command(&["verify".to_string(), input.to_string_lossy().to_string()]);
        assert_eq!(code, 0);
    }

    #[test]
    fn litec_check_rejects_out_of_subset_constructs() {
        let dir = tempdir().expect("tempdir");
        let input = dir.path().join("bad_subset.enk");
        fs::write(
            &input,
            "fn main() -> Int ::\n    let values := [1, 2]\n    for item in values ::\n        print(item)\n    ::\n    return 0\n::\nmain()\n",
        )
        .expect("write input");
        let code = litec_command(&["check".to_string(), input.to_string_lossy().to_string()]);
        assert_ne!(code, 0);
    }

    #[test]
    fn litec_check_accepts_expanded_selfhost_subset() {
        let dir = tempdir().expect("tempdir");
        let input = dir.path().join("expanded_subset.enk");
        fs::write(
            &input,
            "type Counter ::\n    value: Int\n::\n\
             enum Kind ::\n    One\n::\n\
             impl Counter ::\n    fn bump(self: Counter, delta: Int) -> Int ::\n        return self.value + delta\n    ::\n::\n\
             fn main() -> Int ::\n    let add := (x: Int) -> Int => x + 3\n    return add(4)\n::\n\
             main()\n",
        )
        .expect("write input");
        let code = litec_command(&["check".to_string(), input.to_string_lossy().to_string()]);
        assert_eq!(code, 0);
    }

    #[test]
    fn litec_selfhost_verifies_corpus() {
        let dir = tempdir().expect("tempdir");
        let corpus = dir.path().join("corpus");
        fs::create_dir_all(&corpus).expect("corpus dir");
        fs::write(
            corpus.join("a.enk"),
            "fn main() -> Int ::\n    return 7\n::\nmain()\n",
        )
        .expect("write a");
        fs::write(
            corpus.join("b.enk"),
            "type Boxed ::\n    value: Int\n::\n\
             impl Boxed ::\n    fn add(self: Boxed, x: Int) -> Int ::\n        return self.value + x\n    ::\n::\n\
             fn main() -> Int ::\n    let f := (x: Int) -> Int => x + 1\n    return f(6)\n::\n\
             main()\n",
        )
        .expect("write b");
        let code = litec_command(&["selfhost".to_string(), corpus.to_string_lossy().to_string()]);
        assert_eq!(code, 0);
    }

    #[test]
    fn litec_stage_parse_and_codegen() {
        let dir = tempdir().expect("tempdir");
        let input = dir.path().join("stage.enk");
        fs::write(&input, "fn main() -> Int ::\n    return 9\n::\nmain()\n").expect("write input");
        let parse_code = litec_command(&[
            "stage".to_string(),
            "parse".to_string(),
            input.to_string_lossy().to_string(),
        ]);
        assert_eq!(parse_code, 0);
        let out = dir.path().join("stage.bin");
        let codegen_code = litec_command(&[
            "stage".to_string(),
            "codegen".to_string(),
            input.to_string_lossy().to_string(),
            "--out".to_string(),
            out.to_string_lossy().to_string(),
        ]);
        assert_eq!(codegen_code, 0);
        assert!(out.exists());
    }

    #[test]
    fn litec_selfhost_ci_executes_corpus() {
        let dir = tempdir().expect("tempdir");
        let corpus = dir.path().join("ci-corpus");
        fs::create_dir_all(&corpus).expect("corpus dir");
        fs::write(
            corpus.join("main_a.enk"),
            "fn main() -> Int ::\n    let x := 2\n    let f := (v: Int) -> Int => v + 5\n    return f(x)\n::\nmain()\n",
        )
        .expect("write a");
        fs::write(
            corpus.join("main_b.enk"),
            "use std.io\n\
             type Pair ::\n    value: Int\n::\n\
             impl Pair ::\n    fn add(self: Pair, x: Int) -> Int ::\n        return self.value + x\n    ::\n::\n\
             fn main() -> Int ::\n    let p := Pair(4)\n    return 7\n::\n\
             main()\n",
        )
        .expect("write b");
        let code = litec_command(&[
            "selfhost-ci".to_string(),
            corpus.to_string_lossy().to_string(),
        ]);
        assert_eq!(code, 0);
    }
}
