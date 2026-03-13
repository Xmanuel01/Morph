use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::train::{load_config_json, validate_train_config_with_mode};

#[derive(Debug, Clone, Serialize)]
struct ClusterPlan {
    schema_version: u32,
    config_path: String,
    backend: String,
    world_size: usize,
    topology: String,
    rendezvous: String,
    retry_budget: usize,
    device_map: Vec<usize>,
    rank_plans: Vec<RankPlan>,
}

#[derive(Debug, Clone, Serialize)]
struct RankPlan {
    rank: usize,
    device_index: usize,
    env: BTreeMap<String, String>,
    command: Vec<String>,
}

pub fn cluster_command(args: &[String]) -> i32 {
    let Some(subcommand) = args.first().map(|v| v.as_str()) else {
        print_usage();
        return 1;
    };

    match subcommand {
        "validate" => run_validate(&args[1..]),
        "plan" => run_plan(&args[1..]),
        "run" => run_cluster(&args[1..]),
        _ => {
            eprintln!("Unknown cluster subcommand: {}", subcommand);
            print_usage();
            1
        }
    }
}

fn run_validate(args: &[String]) -> i32 {
    let (config_path, json) = match parse_cluster_args(args, false) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("enkai cluster validate: {}", err);
            return 1;
        }
    };

    if let Err(err) = validate_train_config_with_mode(&config_path, true, true) {
        eprintln!("enkai cluster validate: {}", err);
        return 1;
    }
    let plan = match build_cluster_plan(&config_path) {
        Ok(plan) => plan,
        Err(err) => {
            eprintln!("enkai cluster validate: {}", err);
            return 1;
        }
    };

    if json {
        let payload = serde_json::json!({
            "schema_version": 1,
            "ok": true,
            "plan": plan,
        });
        match serde_json::to_string_pretty(&payload) {
            Ok(text) => println!("{}", text),
            Err(err) => {
                eprintln!(
                    "enkai cluster validate: failed to encode JSON output: {}",
                    err
                );
                return 1;
            }
        }
    } else {
        println!(
            "cluster config validated: world_size={} topology={} rendezvous={}",
            plan.world_size, plan.topology, plan.rendezvous
        );
    }
    0
}

fn run_plan(args: &[String]) -> i32 {
    let (config_path, json) = match parse_cluster_args(args, false) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("enkai cluster plan: {}", err);
            return 1;
        }
    };
    let plan = match build_cluster_plan(&config_path) {
        Ok(plan) => plan,
        Err(err) => {
            eprintln!("enkai cluster plan: {}", err);
            return 1;
        }
    };
    if json {
        match serde_json::to_string_pretty(&plan) {
            Ok(text) => println!("{}", text),
            Err(err) => {
                eprintln!("enkai cluster plan: failed to encode JSON output: {}", err);
                return 1;
            }
        }
    } else {
        print_plan(&plan);
    }
    0
}

fn run_cluster(args: &[String]) -> i32 {
    let (config_path, json) = match parse_cluster_args(args, true) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("enkai cluster run: {}", err);
            return 1;
        }
    };
    let plan = match build_cluster_plan(&config_path) {
        Ok(plan) => plan,
        Err(err) => {
            eprintln!("enkai cluster run: {}", err);
            return 1;
        }
    };
    if json {
        match serde_json::to_string_pretty(&plan) {
            Ok(text) => println!("{}", text),
            Err(err) => {
                eprintln!("enkai cluster run: failed to encode JSON output: {}", err);
                return 1;
            }
        }
    } else {
        print_plan(&plan);
    }
    println!(
        "cluster run planner generated launch plan only (operator launcher required for execution)"
    );
    0
}

fn parse_cluster_args(args: &[String], allow_dry_run: bool) -> Result<(PathBuf, bool), String> {
    let mut json = false;
    let mut config_path: Option<PathBuf> = None;
    for arg in args {
        match arg.as_str() {
            "--json" => json = true,
            "--dry-run" if allow_dry_run => {}
            _ if arg.starts_with("--") => {
                return Err(format!(
                    "unknown option {} (supported: --json{})",
                    arg,
                    if allow_dry_run { ", --dry-run" } else { "" }
                ));
            }
            _ => {
                if config_path.is_some() {
                    return Err("expected exactly one config file path".to_string());
                }
                config_path = Some(PathBuf::from(arg));
            }
        }
    }
    let Some(path) = config_path else {
        return Err("missing config file path".to_string());
    };
    Ok((path, json))
}

fn build_cluster_plan(config_path: &Path) -> Result<ClusterPlan, String> {
    let config = load_config_json(config_path)?;
    let root = config
        .as_object()
        .ok_or_else(|| "config root must be a JSON object".to_string())?;

    let backend = root
        .get("backend")
        .and_then(|v| v.as_str())
        .unwrap_or("cpu")
        .to_string();
    let world_size = json_usize_with_default(root, "world_size", 1)?;
    let rank = json_usize_with_default(root, "rank", 0)?;
    if world_size == 0 {
        return Err("world_size must be >= 1".to_string());
    }
    if rank >= world_size {
        return Err("rank must be < world_size".to_string());
    }

    let dist = root.get("dist").and_then(|v| v.as_object());
    let mut topology = if world_size > 1 {
        "single-node".to_string()
    } else {
        "standalone".to_string()
    };
    let mut rendezvous = "env://".to_string();
    let mut retry_budget = 3usize;
    let mut device_map: Vec<usize> = (0..world_size).collect();

    if let Some(dist_obj) = dist {
        if let Some(value) = dist_obj.get("topology").and_then(|v| v.as_str()) {
            topology = value.to_ascii_lowercase();
        }
        if let Some(value) = dist_obj.get("rendezvous").and_then(|v| v.as_str()) {
            rendezvous = value.to_string();
        }
        if let Some(value) = dist_obj.get("retry_budget") {
            retry_budget = json_usize(value, "dist.retry_budget")?;
        }
        if let Some(value) = dist_obj.get("device_map") {
            device_map = parse_device_map_value(value, world_size)?;
        }
    }

    if world_size > 1 {
        if topology != "single-node" && topology != "multi-node" {
            return Err(format!(
                "dist.topology must be \"single-node\" or \"multi-node\" (found {})",
                topology
            ));
        }
        if topology == "multi-node" && rendezvous.eq_ignore_ascii_case("env://") {
            return Err(
                "multi-node topology requires dist.rendezvous=tcp://<host>:<port>".to_string(),
            );
        }
    } else {
        topology = "standalone".to_string();
    }

    if device_map.len() != world_size {
        return Err(format!(
            "dist.device_map length mismatch: expected {}, found {}",
            world_size,
            device_map.len()
        ));
    }
    let mut seen = std::collections::BTreeSet::new();
    for (idx, value) in device_map.iter().enumerate() {
        if *value >= world_size {
            return Err(format!(
                "dist.device_map[{}] out of range: {} >= {}",
                idx, value, world_size
            ));
        }
        if !seen.insert(*value) {
            return Err(format!(
                "dist.device_map duplicate CUDA index {} (must be one-to-one)",
                value
            ));
        }
    }

    let mut rank_plans = Vec::with_capacity(world_size);
    for (rank_idx, device_index) in device_map.iter().enumerate().take(world_size) {
        let mut env = BTreeMap::new();
        env.insert("ENKAI_ENABLE_DIST".to_string(), "1".to_string());
        env.insert("WORLD_SIZE".to_string(), world_size.to_string());
        env.insert("RANK".to_string(), rank_idx.to_string());
        env.insert("ENKAI_DIST_TOPOLOGY".to_string(), topology.clone());
        env.insert("ENKAI_DIST_RENDEZVOUS".to_string(), rendezvous.clone());
        env.insert(
            "ENKAI_DIST_RETRY_BUDGET".to_string(),
            retry_budget.to_string(),
        );
        env.insert(
            "ENKAI_DIST_DEVICE_INDEX".to_string(),
            device_index.to_string(),
        );
        rank_plans.push(RankPlan {
            rank: rank_idx,
            device_index: *device_index,
            env,
            command: vec![
                "enkai".to_string(),
                "train".to_string(),
                config_path.to_string_lossy().to_string(),
                "--strict-contracts".to_string(),
            ],
        });
    }

    Ok(ClusterPlan {
        schema_version: 1,
        config_path: config_path.to_string_lossy().to_string(),
        backend,
        world_size,
        topology,
        rendezvous,
        retry_budget,
        device_map,
        rank_plans,
    })
}

fn json_usize_with_default(
    map: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: usize,
) -> Result<usize, String> {
    let Some(value) = map.get(key) else {
        return Ok(default);
    };
    json_usize(value, key)
}

fn json_usize(value: &serde_json::Value, key: &str) -> Result<usize, String> {
    match value {
        serde_json::Value::Number(v) if v.is_u64() => Ok(v.as_u64().unwrap_or_default() as usize),
        serde_json::Value::Number(v) if v.is_i64() && v.as_i64().unwrap_or_default() >= 0 => {
            Ok(v.as_i64().unwrap_or_default() as usize)
        }
        _ => Err(format!("{} must be an integer >= 0", key)),
    }
}

fn parse_device_map_value(
    value: &serde_json::Value,
    world_size: usize,
) -> Result<Vec<usize>, String> {
    match value {
        serde_json::Value::String(text) => parse_device_map_string(text, world_size),
        serde_json::Value::Array(values) => values
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                json_usize(item, "dist.device_map")
                    .map_err(|err| format!("{} at index {}", err, idx))
            })
            .collect(),
        _ => Err("dist.device_map must be String or Array[Int]".to_string()),
    }
}

fn parse_device_map_string(value: &str, world_size: usize) -> Result<Vec<usize>, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok((0..world_size.max(1)).collect());
    }
    trimmed
        .split(',')
        .map(|part| {
            let token = part.trim();
            token.parse::<usize>().map_err(|_| {
                format!(
                    "dist.device_map contains invalid value {:?}; expected comma-separated integers",
                    token
                )
            })
        })
        .collect()
}

fn print_plan(plan: &ClusterPlan) {
    println!(
        "cluster plan: world_size={} topology={} rendezvous={} retry_budget={}",
        plan.world_size, plan.topology, plan.rendezvous, plan.retry_budget
    );
    for rank in &plan.rank_plans {
        println!(
            "  rank {} -> cuda:{} | env: WORLD_SIZE={} RANK={}",
            rank.rank,
            rank.device_index,
            rank.env.get("WORLD_SIZE").map(String::as_str).unwrap_or(""),
            rank.env.get("RANK").map(String::as_str).unwrap_or(""),
        );
    }
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  enkai cluster validate <config.enk> [--json]");
    eprintln!("  enkai cluster plan <config.enk> [--json]");
    eprintln!("  enkai cluster run <config.enk> [--dry-run] [--json]");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use tempfile::tempdir;

    #[test]
    fn parse_device_map_string_accepts_csv() {
        let parsed = parse_device_map_string("0,2,1", 3).expect("parse");
        assert_eq!(parsed, vec![0, 2, 1]);
    }

    #[test]
    fn parse_device_map_string_rejects_invalid_values() {
        let err = parse_device_map_string("0,abc,1", 3).expect_err("must fail");
        assert!(err.contains("invalid value"));
    }

    #[test]
    fn build_cluster_plan_supports_multi_node_dist_contract() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("config.enk");
        let payload = serde_json::json!({
            "backend": "native",
            "world_size": 2,
            "rank": 0,
            "dist": {
                "topology": "multi-node",
                "rendezvous": "tcp://127.0.0.1:29500",
                "retry_budget": 7,
                "device_map": "1,0"
            }
        });
        let source = format!(
            "fn main() ::\n    return json.parse(\"{}\")\n::\n",
            payload
                .to_string()
                .replace('\\', "\\\\")
                .replace('\"', "\\\"")
        );
        fs::write(&config_path, source).expect("config");
        let plan = build_cluster_plan(&config_path).expect("plan");
        assert_eq!(plan.world_size, 2);
        assert_eq!(plan.topology, "multi-node");
        assert_eq!(plan.rendezvous, "tcp://127.0.0.1:29500");
        assert_eq!(plan.retry_budget, 7);
        assert_eq!(plan.device_map, vec![1, 0]);
        assert_eq!(plan.rank_plans[0].device_index, 1);
        assert_eq!(plan.rank_plans[1].device_index, 0);
    }

    #[test]
    fn build_cluster_plan_rejects_multi_node_env_rendezvous() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("config.enk");
        let payload = serde_json::json!({
            "backend": "native",
            "world_size": 2,
            "rank": 0,
            "dist": {
                "topology": "multi-node",
                "rendezvous": "env://",
                "device_map": "0,1"
            }
        });
        let source = format!(
            "fn main() ::\n    return json.parse(\"{}\")\n::\n",
            payload
                .to_string()
                .replace('\\', "\\\\")
                .replace('\"', "\\\"")
        );
        fs::write(&config_path, source).expect("config");
        let err = build_cluster_plan(&config_path).expect_err("must fail");
        assert!(err.contains("multi-node topology requires"));
    }
}
