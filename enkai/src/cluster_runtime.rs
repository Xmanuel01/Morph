use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::cluster::{
    ClusterArgs, ClusterPlan, ClusterRunReport, RankExecutionReport, RankPlan,
    SimulationExecutionSummary, SimulationPlan, TrainClusterContract,
};
use crate::systems::ClusterCommandManifest;
use crate::train::{load_config_json, validate_train_config_with_mode};

pub(crate) fn execute_manifest(manifest: &ClusterCommandManifest) -> i32 {
    match manifest.subcommand.as_str() {
        "validate" => execute_validate_manifest(manifest),
        "plan" => execute_plan_manifest(manifest),
        "run" => execute_run_manifest(manifest),
        other => {
            eprintln!("enkai cluster: unknown manifest subcommand '{}'", other);
            1
        }
    }
}

pub(crate) fn build_cluster_plan(config_path: &Path) -> Result<ClusterPlan, String> {
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
    let mut hosts = vec!["localhost".to_string()];
    let mut host_map = vec![0usize; world_size];

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
        if let Some(value) = dist_obj.get("hosts") {
            hosts = parse_hosts_value(value)?;
        }
        if let Some(value) = dist_obj.get("host_map") {
            host_map = parse_host_map_value(value, world_size)?;
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
    let mut seen = BTreeSet::new();
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

    if topology == "multi-node" {
        if hosts.is_empty() {
            return Err(
                "dist.hosts must define at least one host for multi-node topology".to_string(),
            );
        }
    } else {
        hosts = vec!["localhost".to_string()];
        host_map = vec![0usize; world_size];
    }

    if host_map.len() != world_size {
        return Err(format!(
            "dist.host_map length mismatch: expected {}, found {}",
            world_size,
            host_map.len()
        ));
    }
    for (idx, host_index) in host_map.iter().enumerate() {
        if *host_index >= hosts.len() {
            return Err(format!(
                "dist.host_map[{}] out of range: {} >= {}",
                idx,
                host_index,
                hosts.len()
            ));
        }
    }

    let workload_kind = root
        .get("workload")
        .and_then(|v| v.as_str())
        .unwrap_or("train")
        .to_ascii_lowercase();
    let checkpoint_policy = root.get("checkpoint_policy").and_then(|v| v.as_object());
    let validate_on_resume = checkpoint_policy
        .and_then(|obj| obj.get("validate_on_resume"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let train_contract = if workload_kind == "train" {
        Some(TrainClusterContract {
            strict_contracts: true,
            validate_on_resume,
        })
    } else {
        None
    };

    let config_dir = config_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let simulation = if workload_kind == "simulation" {
        let sim_obj = root
            .get("simulation")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "simulation workload requires simulation object".to_string())?;
        let target_text = sim_obj
            .get("target")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "simulation.target must be a path string".to_string())?;
        let target_path = resolve_config_relative_path(&config_dir, target_text);
        let total_steps = json_usize_with_default(sim_obj, "total_steps", 1)?;
        if total_steps == 0 {
            return Err("simulation.total_steps must be >= 1".to_string());
        }
        let step_window = json_usize_with_default(sim_obj, "step_window", total_steps.min(16))?;
        if step_window == 0 {
            return Err("simulation.step_window must be >= 1".to_string());
        }
        let snapshot_interval = json_usize_with_default(sim_obj, "snapshot_interval", step_window)?;
        if snapshot_interval == 0 {
            return Err("simulation.snapshot_interval must be >= 1".to_string());
        }
        let partition_count = json_usize_with_default(sim_obj, "partition_count", world_size)?;
        if partition_count != world_size {
            return Err(format!(
                "simulation.partition_count must equal world_size for v2.9.4 (expected {}, found {})",
                world_size, partition_count
            ));
        }
        let recovery_dir = sim_obj
            .get("recovery_dir")
            .and_then(|v| v.as_str())
            .map(|value| resolve_config_relative_path(&config_dir, value))
            .unwrap_or_else(|| config_dir.join("recovery"));
        let route_policy = sim_obj
            .get("route_policy")
            .and_then(|v| v.as_str())
            .unwrap_or("deterministic-ring")
            .to_ascii_lowercase();
        if !matches!(
            route_policy.as_str(),
            "deterministic-ring" | "broadcast-local" | "shard-hash"
        ) {
            return Err(format!(
                "simulation.route_policy must be deterministic-ring|broadcast-local|shard-hash (found {})",
                route_policy
            ));
        }
        let seed = sim_obj.get("seed").and_then(|v| v.as_u64()).unwrap_or(0);
        Some(SimulationPlan {
            target: target_path.to_string_lossy().to_string(),
            partition_count,
            total_steps,
            step_window,
            snapshot_interval,
            recovery_dir: recovery_dir.to_string_lossy().to_string(),
            route_policy,
            seed,
        })
    } else if workload_kind == "train" {
        None
    } else {
        return Err(format!(
            "workload must be \"train\" or \"simulation\" (found {})",
            workload_kind
        ));
    };

    let mut rank_plans = Vec::with_capacity(world_size);
    for rank_idx in 0..world_size {
        let host_index = host_map[rank_idx];
        let host = hosts
            .get(host_index)
            .cloned()
            .unwrap_or_else(|| "localhost".to_string());
        let device_index = device_map[rank_idx];
        let mut env_map = BTreeMap::new();
        env_map.insert("ENKAI_CLUSTER_RANK".to_string(), rank_idx.to_string());
        env_map.insert(
            "ENKAI_CLUSTER_WORLD_SIZE".to_string(),
            world_size.to_string(),
        );
        env_map.insert("ENKAI_CLUSTER_HOST".to_string(), host.clone());
        env_map.insert(
            "ENKAI_CLUSTER_HOST_INDEX".to_string(),
            host_index.to_string(),
        );
        env_map.insert(
            "ENKAI_CLUSTER_DEVICE_INDEX".to_string(),
            device_index.to_string(),
        );
        env_map.insert("ENKAI_CLUSTER_RENDEZVOUS".to_string(), rendezvous.clone());
        env_map.insert("ENKAI_CLUSTER_TOPOLOGY".to_string(), topology.clone());
        env_map.insert("ENKAI_CLUSTER_WORKLOAD".to_string(), workload_kind.clone());
        let command = if let Some(sim_plan) = &simulation {
            env_map.insert("ENKAI_SIM_PARTITION_ID".to_string(), rank_idx.to_string());
            env_map.insert(
                "ENKAI_SIM_PARTITION_COUNT".to_string(),
                sim_plan.partition_count.to_string(),
            );
            env_map.insert(
                "ENKAI_SIM_TOTAL_STEPS".to_string(),
                sim_plan.total_steps.to_string(),
            );
            env_map.insert(
                "ENKAI_SIM_STEP_WINDOW".to_string(),
                sim_plan.step_window.to_string(),
            );
            env_map.insert(
                "ENKAI_SIM_SNAPSHOT_INTERVAL".to_string(),
                sim_plan.snapshot_interval.to_string(),
            );
            env_map.insert(
                "ENKAI_SIM_ROUTE_POLICY".to_string(),
                sim_plan.route_policy.clone(),
            );
            env_map.insert("ENKAI_SIM_SEED".to_string(), sim_plan.seed.to_string());
            env_map.insert(
                "ENKAI_SIM_RECOVERY_DIR".to_string(),
                Path::new(&sim_plan.recovery_dir)
                    .join(format!("rank{}", rank_idx))
                    .to_string_lossy()
                    .to_string(),
            );
            let next_rank = if sim_plan.partition_count <= 1 {
                rank_idx
            } else {
                (rank_idx + 1) % sim_plan.partition_count
            };
            env_map.insert(
                "ENKAI_SIM_ROUTE_NEXT_RANK".to_string(),
                next_rank.to_string(),
            );
            vec![
                "enkai".to_string(),
                "sim".to_string(),
                sim_plan.target.clone(),
            ]
        } else {
            vec![
                "enkai".to_string(),
                "train".to_string(),
                config_path.to_string_lossy().to_string(),
            ]
        };
        rank_plans.push(RankPlan {
            rank: rank_idx,
            host_index,
            host,
            device_index,
            env: env_map,
            command,
        });
    }

    Ok(ClusterPlan {
        schema_version: 1,
        config_path: config_path.to_string_lossy().to_string(),
        backend,
        workload_kind,
        world_size,
        topology,
        rendezvous,
        retry_budget,
        device_map,
        hosts,
        host_map,
        train_contract,
        simulation,
        rank_plans,
    })
}

pub(crate) fn cluster_args_from_manifest(manifest: &ClusterCommandManifest) -> ClusterArgs {
    ClusterArgs {
        config_path: PathBuf::from(&manifest.config_path),
        json: manifest.json,
        dry_run: manifest.dry_run,
        output: manifest.output.as_ref().map(PathBuf::from),
    }
}

pub(crate) fn execute_validate_manifest(manifest: &ClusterCommandManifest) -> i32 {
    let options = cluster_args_from_manifest(manifest);

    if let Err(err) = validate_train_config_with_mode(&options.config_path, true, true) {
        eprintln!("enkai cluster validate: {}", err);
        return 1;
    }
    let plan = match build_cluster_plan(&options.config_path) {
        Ok(plan) => plan,
        Err(err) => {
            eprintln!("enkai cluster validate: {}", err);
            return 1;
        }
    };

    let payload = serde_json::json!({
        "schema_version": 1,
        "ok": true,
        "plan": plan,
    });
    if let Err(err) = emit_command_output(&payload, options.json, options.output.as_deref()) {
        eprintln!("enkai cluster validate: {}", err);
        return 1;
    }
    if !options.json {
        println!(
            "cluster config validated: world_size={} topology={} rendezvous={} workload={}",
            payload["plan"]["world_size"],
            payload["plan"]["topology"],
            payload["plan"]["rendezvous"],
            payload["plan"]["workload_kind"]
        );
    }
    0
}

pub(crate) fn execute_plan_manifest(manifest: &ClusterCommandManifest) -> i32 {
    let options = cluster_args_from_manifest(manifest);
    let plan = match build_cluster_plan(&options.config_path) {
        Ok(plan) => plan,
        Err(err) => {
            eprintln!("enkai cluster plan: {}", err);
            return 1;
        }
    };
    if let Err(err) = emit_command_output(&plan, options.json, options.output.as_deref()) {
        eprintln!("enkai cluster plan: {}", err);
        return 1;
    }
    if !options.json {
        print_plan(&plan);
    }
    0
}

pub(crate) fn execute_run_manifest(manifest: &ClusterCommandManifest) -> i32 {
    let options = cluster_args_from_manifest(manifest);
    let plan = match build_cluster_plan(&options.config_path) {
        Ok(plan) => plan,
        Err(err) => {
            eprintln!("enkai cluster run: {}", err);
            return 1;
        }
    };

    let report = if options.dry_run {
        planner_only_report(&plan, "dry_run")
    } else if plan.workload_kind == "simulation" {
        match execute_simulation_cluster(&plan) {
            Ok(report) => report,
            Err(err) => {
                eprintln!("enkai cluster run: {}", err);
                return 1;
            }
        }
    } else {
        planner_only_report(
            &plan,
            "train multi-node execution remains operator-managed; use plan output for launch supervision",
        )
    };

    if let Err(err) = emit_command_output(&report, options.json, options.output.as_deref()) {
        eprintln!("enkai cluster run: {}", err);
        return 1;
    }

    if !options.json {
        print_run_report(&report);
    }
    if report.all_passed {
        0
    } else {
        1
    }
}

pub(crate) fn build_cluster_plan_json(config_path: &Path) -> Result<serde_json::Value, String> {
    let plan = build_cluster_plan(config_path)?;
    serde_json::to_value(&plan).map_err(|err| format!("serialize cluster plan: {}", err))
}

pub(crate) fn execute_simulation_cluster(plan: &ClusterPlan) -> Result<ClusterRunReport, String> {
    let simulation = plan
        .simulation
        .as_ref()
        .ok_or_else(|| "simulation plan missing".to_string())?;
    let mut rank_reports = Vec::with_capacity(plan.rank_plans.len());
    let mut all_passed = true;
    let mut resumed_partitions = 0usize;
    let mut total_completed_steps = 0usize;

    for rank_plan in &plan.rank_plans {
        let report = execute_simulation_rank(simulation, rank_plan, plan.retry_budget)?;
        if report.recovered_from_snapshot {
            resumed_partitions += 1;
        }
        total_completed_steps += report.completed_steps;
        if report.status != "ok" {
            all_passed = false;
        }
        rank_reports.push(report);
    }

    Ok(ClusterRunReport {
        schema_version: 1,
        workload_kind: "simulation".to_string(),
        execution_mode: "supervised".to_string(),
        config_path: plan.config_path.clone(),
        topology: plan.topology.clone(),
        world_size: plan.world_size,
        retry_budget: plan.retry_budget,
        all_passed,
        plan_only_reason: None,
        simulation: Some(SimulationExecutionSummary {
            target: simulation.target.clone(),
            partition_count: simulation.partition_count,
            total_steps: simulation.total_steps,
            step_window: simulation.step_window,
            snapshot_interval: simulation.snapshot_interval,
            route_policy: simulation.route_policy.clone(),
            recovery_dir: simulation.recovery_dir.clone(),
            resumed_partitions,
            total_completed_steps,
        }),
        rank_reports,
    })
}

fn execute_simulation_rank(
    simulation: &SimulationPlan,
    rank_plan: &RankPlan,
    retry_budget: usize,
) -> Result<RankExecutionReport, String> {
    let artifact_dir =
        PathBuf::from(&simulation.recovery_dir).join(format!("rank{}", rank_plan.rank));
    fs::create_dir_all(&artifact_dir)
        .map_err(|err| format!("failed to create {}: {}", artifact_dir.display(), err))?;

    let mut attempts = 0usize;
    let mut retries_used = 0usize;
    let mut completed_steps = 0usize;
    let mut windows_completed = 0usize;
    let mut recovered_from_snapshot = false;
    let mut current_snapshot: Option<PathBuf> = None;
    let mut status = "ok".to_string();
    let mut exit_code = 0i32;
    let mut window_reports = Vec::new();
    let mut stdout_logs = Vec::new();
    let mut stderr_logs = Vec::new();

    while completed_steps < simulation.total_steps {
        let window_index = windows_completed;
        let window_prefix = artifact_dir.join(format!("window_{:04}", window_index));
        let report_path = if current_snapshot.is_some() {
            window_prefix.with_extension("replay.json")
        } else {
            window_prefix.with_extension("run.json")
        };
        let lineage_path = window_prefix.with_extension("lineage.json");
        let snapshot_manifest_path = window_prefix.with_extension("snapshot.manifest.json");
        let snapshot_path = window_prefix.with_extension("snapshot.json");
        let stdout_path = window_prefix.with_extension("stdout.log");
        let stderr_path = window_prefix.with_extension("stderr.log");
        let step_chunk = simulation
            .step_window
            .min(simulation.total_steps.saturating_sub(completed_steps))
            .max(1);
        let mut child_env = rank_plan.env.clone();
        child_env.insert("ENKAI_SIM_STEP_BUDGET".to_string(), step_chunk.to_string());

        let execution = if let Some(snapshot) = &current_snapshot {
            crate::cluster_sim_runtime::SimExecution::Replay(crate::sim::SimReplayOptions {
                snapshot: snapshot.clone(),
                steps: step_chunk,
                emit_json_stdout: false,
                output: Some(report_path.clone()),
                snapshot_output: Some(snapshot_path.clone()),
                lineage_output: Some(lineage_path.clone()),
                snapshot_manifest_output: Some(snapshot_manifest_path.clone()),
            })
        } else {
            crate::cluster_sim_runtime::SimExecution::Run(crate::sim::SimRunOptions {
                target: PathBuf::from(&simulation.target),
                trace_vm: false,
                disasm: false,
                trace_task: false,
                trace_net: false,
                emit_json_stdout: false,
                output: Some(report_path.clone()),
                snapshot_output: Some(snapshot_path.clone()),
                lineage_output: Some(lineage_path.clone()),
                snapshot_manifest_output: Some(snapshot_manifest_path.clone()),
            })
        };

        let mut result = crate::cluster_sim_runtime::run_sim_execution_with_logs(
            &execution,
            &child_env,
            &stdout_path,
            &stderr_path,
        )?;
        if result == 0
            && current_snapshot.is_some()
            && crate::cluster_sim_runtime::should_inject_retry_failure(
                rank_plan.rank,
                windows_completed,
            )
        {
            let note = format!(
                "injected cluster retry fault for rank {} after {} completed windows\n",
                rank_plan.rank, windows_completed
            );
            let _ = fs::write(&stderr_path, note);
            let _ = fs::remove_file(&snapshot_path);
            result = fault_injection_exit_code();
        }
        attempts += 1;
        window_reports.push(report_path.to_string_lossy().to_string());
        stdout_logs.push(stdout_path.to_string_lossy().to_string());
        stderr_logs.push(stderr_path.to_string_lossy().to_string());
        exit_code = result;

        if result == 0 {
            if !snapshot_path.is_file() {
                return Err(format!(
                    "simulation rank {} did not produce snapshot {}",
                    rank_plan.rank,
                    snapshot_path.display()
                ));
            }
            current_snapshot = Some(snapshot_path);
            completed_steps += step_chunk;
            windows_completed += 1;
            continue;
        }

        if retries_used < retry_budget {
            retries_used += 1;
            recovered_from_snapshot |= current_snapshot.is_some();
            continue;
        }

        status = "failed".to_string();
        break;
    }

    if completed_steps >= simulation.total_steps {
        status = "ok".to_string();
    }

    Ok(RankExecutionReport {
        rank: rank_plan.rank,
        host: rank_plan.host.clone(),
        attempts,
        retries_used,
        completed_steps,
        windows_completed,
        recovered_from_snapshot,
        status,
        exit_code,
        artifact_dir: artifact_dir.to_string_lossy().to_string(),
        final_snapshot: current_snapshot.map(|path| path.to_string_lossy().to_string()),
        window_reports,
        stdout_logs,
        stderr_logs,
    })
}

pub(crate) fn json_usize_with_default(
    map: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: usize,
) -> Result<usize, String> {
    let Some(value) = map.get(key) else {
        return Ok(default);
    };
    json_usize(value, key)
}

pub(crate) fn json_usize(value: &serde_json::Value, key: &str) -> Result<usize, String> {
    match value {
        serde_json::Value::Number(v) if v.is_u64() => Ok(v.as_u64().unwrap_or_default() as usize),
        serde_json::Value::Number(v) if v.is_i64() && v.as_i64().unwrap_or_default() >= 0 => {
            Ok(v.as_i64().unwrap_or_default() as usize)
        }
        _ => Err(format!("{} must be an integer >= 0", key)),
    }
}

pub(crate) fn parse_device_map_value(
    value: &serde_json::Value,
    world_size: usize,
) -> Result<Vec<usize>, String> {
    match value {
        serde_json::Value::String(text) => parse_usize_csv(text, world_size, "dist.device_map"),
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

pub(crate) fn parse_host_map_value(
    value: &serde_json::Value,
    world_size: usize,
) -> Result<Vec<usize>, String> {
    match value {
        serde_json::Value::String(text) => parse_usize_csv(text, world_size, "dist.host_map"),
        serde_json::Value::Array(values) => values
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                json_usize(item, "dist.host_map").map_err(|err| format!("{} at index {}", err, idx))
            })
            .collect(),
        _ => Err("dist.host_map must be String or Array[Int]".to_string()),
    }
}

pub(crate) fn parse_hosts_value(value: &serde_json::Value) -> Result<Vec<String>, String> {
    match value {
        serde_json::Value::String(text) => {
            let hosts = text
                .split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>();
            if hosts.is_empty() {
                return Err("dist.hosts must not be empty".to_string());
            }
            Ok(hosts)
        }
        serde_json::Value::Array(values) => {
            let mut hosts = Vec::with_capacity(values.len());
            for (idx, value) in values.iter().enumerate() {
                let host = value
                    .as_str()
                    .ok_or_else(|| format!("dist.hosts[{}] must be a non-empty string", idx))?;
                let trimmed = host.trim();
                if trimmed.is_empty() {
                    return Err(format!("dist.hosts[{}] must be a non-empty string", idx));
                }
                hosts.push(trimmed.to_string());
            }
            if hosts.is_empty() {
                return Err("dist.hosts must not be empty".to_string());
            }
            Ok(hosts)
        }
        _ => Err("dist.hosts must be String or Array[String]".to_string()),
    }
}

pub(crate) fn parse_usize_csv(
    value: &str,
    world_size: usize,
    key: &str,
) -> Result<Vec<usize>, String> {
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
                    "{} contains invalid value {:?}; expected comma-separated integers",
                    key, token
                )
            })
        })
        .collect()
}

pub(crate) fn resolve_config_relative_path(base_dir: &Path, value: &str) -> PathBuf {
    let candidate = PathBuf::from(value);
    if candidate.is_absolute() {
        candidate
    } else {
        base_dir.join(candidate)
    }
}

pub(crate) fn planner_only_report(plan: &ClusterPlan, reason: &str) -> ClusterRunReport {
    ClusterRunReport {
        schema_version: 1,
        workload_kind: plan.workload_kind.clone(),
        execution_mode: "plan_only".to_string(),
        config_path: plan.config_path.clone(),
        topology: plan.topology.clone(),
        world_size: plan.world_size,
        retry_budget: plan.retry_budget,
        all_passed: true,
        plan_only_reason: Some(reason.to_string()),
        simulation: plan
            .simulation
            .as_ref()
            .map(|sim| SimulationExecutionSummary {
                target: sim.target.clone(),
                partition_count: sim.partition_count,
                total_steps: sim.total_steps,
                step_window: sim.step_window,
                snapshot_interval: sim.snapshot_interval,
                route_policy: sim.route_policy.clone(),
                recovery_dir: sim.recovery_dir.clone(),
                resumed_partitions: 0,
                total_completed_steps: 0,
            }),
        rank_reports: plan
            .rank_plans
            .iter()
            .map(|rank| RankExecutionReport {
                rank: rank.rank,
                host: rank.host.clone(),
                attempts: 0,
                retries_used: 0,
                completed_steps: 0,
                windows_completed: 0,
                recovered_from_snapshot: false,
                status: "planned".to_string(),
                exit_code: 0,
                artifact_dir: String::new(),
                final_snapshot: None,
                window_reports: Vec::new(),
                stdout_logs: Vec::new(),
                stderr_logs: Vec::new(),
            })
            .collect(),
    }
}

pub(crate) fn fault_injection_exit_code() -> i32 {
    env::var("ENKAI_CLUSTER_INJECT_FAIL_ONCE_EXIT_CODE")
        .ok()
        .and_then(|value| value.trim().parse::<i32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(75)
}

pub(crate) fn emit_command_output<T: Serialize>(
    payload: &T,
    emit_stdout_json: bool,
    output: Option<&Path>,
) -> Result<(), String> {
    let text = serde_json::to_string_pretty(payload).map_err(|err| err.to_string())?;
    if emit_stdout_json {
        println!("{}", text);
    }
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|err| err.to_string())?;
            }
        }
        fs::write(path, text).map_err(|err| err.to_string())?;
    }
    Ok(())
}

pub(crate) fn print_plan(plan: &ClusterPlan) {
    println!(
        "cluster plan: workload={} world_size={} topology={} rendezvous={} retry_budget={}",
        plan.workload_kind, plan.world_size, plan.topology, plan.rendezvous, plan.retry_budget
    );
    if let Some(simulation) = &plan.simulation {
        println!(
            "  simulation target={} steps={} window={} route_policy={} recovery_dir={}",
            simulation.target,
            simulation.total_steps,
            simulation.step_window,
            simulation.route_policy,
            simulation.recovery_dir
        );
    }
    for rank in &plan.rank_plans {
        println!(
            "  rank {} -> host={} cuda:{} | env: WORLD_SIZE={} RANK={}",
            rank.rank,
            rank.host,
            rank.device_index,
            rank.env.get("WORLD_SIZE").map(String::as_str).unwrap_or(""),
            rank.env.get("RANK").map(String::as_str).unwrap_or(""),
        );
    }
}

pub(crate) fn print_run_report(report: &ClusterRunReport) {
    println!(
        "cluster run: workload={} mode={} all_passed={} topology={} world_size={}",
        report.workload_kind,
        report.execution_mode,
        report.all_passed,
        report.topology,
        report.world_size
    );
    if let Some(reason) = &report.plan_only_reason {
        println!("  note: {}", reason);
    }
    for rank in &report.rank_reports {
        println!(
            "  rank {} host={} status={} attempts={} steps={} snapshot={}",
            rank.rank,
            rank.host,
            rank.status,
            rank.attempts,
            rank.completed_steps,
            rank.final_snapshot.as_deref().unwrap_or("-"),
        );
    }
}
