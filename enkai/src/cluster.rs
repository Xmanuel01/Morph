use std::collections::BTreeMap;
#[cfg(test)]
use std::fs;
use std::path::PathBuf;

use serde::Serialize;

use crate::systems::ClusterCommandManifest;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ClusterPlan {
    pub(crate) schema_version: u32,
    pub(crate) config_path: String,
    pub(crate) backend: String,
    pub(crate) workload_kind: String,
    pub(crate) world_size: usize,
    pub(crate) topology: String,
    pub(crate) rendezvous: String,
    pub(crate) retry_budget: usize,
    pub(crate) device_map: Vec<usize>,
    pub(crate) hosts: Vec<String>,
    pub(crate) host_map: Vec<usize>,
    pub(crate) train_contract: Option<TrainClusterContract>,
    pub(crate) simulation: Option<SimulationPlan>,
    pub(crate) rank_plans: Vec<RankPlan>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TrainClusterContract {
    pub(crate) strict_contracts: bool,
    pub(crate) validate_on_resume: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SimulationPlan {
    pub(crate) target: String,
    pub(crate) partition_count: usize,
    pub(crate) total_steps: usize,
    pub(crate) step_window: usize,
    pub(crate) snapshot_interval: usize,
    pub(crate) recovery_dir: String,
    pub(crate) route_policy: String,
    pub(crate) seed: u64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RankPlan {
    pub(crate) rank: usize,
    pub(crate) host_index: usize,
    pub(crate) host: String,
    pub(crate) device_index: usize,
    pub(crate) env: BTreeMap<String, String>,
    pub(crate) command: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ClusterArgs {
    pub(crate) config_path: PathBuf,
    pub(crate) json: bool,
    pub(crate) dry_run: bool,
    pub(crate) output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ClusterRunReport {
    pub(crate) schema_version: u32,
    pub(crate) workload_kind: String,
    pub(crate) execution_mode: String,
    pub(crate) config_path: String,
    pub(crate) topology: String,
    pub(crate) world_size: usize,
    pub(crate) retry_budget: usize,
    pub(crate) all_passed: bool,
    pub(crate) plan_only_reason: Option<String>,
    pub(crate) simulation: Option<SimulationExecutionSummary>,
    pub(crate) rank_reports: Vec<RankExecutionReport>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SimulationExecutionSummary {
    pub(crate) target: String,
    pub(crate) partition_count: usize,
    pub(crate) total_steps: usize,
    pub(crate) step_window: usize,
    pub(crate) snapshot_interval: usize,
    pub(crate) route_policy: String,
    pub(crate) recovery_dir: String,
    pub(crate) resumed_partitions: usize,
    pub(crate) total_completed_steps: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RankExecutionReport {
    pub(crate) rank: usize,
    pub(crate) host: String,
    pub(crate) attempts: usize,
    pub(crate) retries_used: usize,
    pub(crate) completed_steps: usize,
    pub(crate) windows_completed: usize,
    pub(crate) recovered_from_snapshot: bool,
    pub(crate) status: String,
    pub(crate) exit_code: i32,
    pub(crate) artifact_dir: String,
    pub(crate) final_snapshot: Option<String>,
    pub(crate) window_reports: Vec<String>,
    pub(crate) stdout_logs: Vec<String>,
    pub(crate) stderr_logs: Vec<String>,
}

pub fn cluster_command(args: &[String]) -> i32 {
    let Some(subcommand) = args.first().map(|v| v.as_str()) else {
        print_usage();
        return 1;
    };

    match subcommand {
        "validate" | "plan" | "run" => {
            let manifest = match crate::systems::build_cluster_command_manifest(args) {
                Ok(value) => value,
                Err(err) => {
                    eprintln!("enkai cluster {}: {}", subcommand, err);
                    return 1;
                }
            };
            execute_cluster_manifest(&manifest)
        }
        _ => {
            eprintln!("Unknown cluster subcommand: {}", subcommand);
            print_usage();
            1
        }
    }
}

pub(crate) fn execute_cluster_manifest(manifest: &ClusterCommandManifest) -> i32 {
    crate::cluster_runtime::execute_manifest(manifest)
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  enkai cluster validate <config.enk> [--json] [--output <file>]");
    eprintln!("  enkai cluster plan <config.enk> [--json] [--output <file>]");
    eprintln!("  enkai cluster run <config.enk> [--dry-run] [--json] [--output <file>]");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use tempfile::tempdir_in;

    fn write_cluster_config(dir: &Path, body: &str) -> PathBuf {
        let config = dir.join("cluster_config.enk");
        fs::write(&config, body).expect("write config");
        config
    }

    #[test]
    fn parse_cluster_args_supports_output_and_dry_run() {
        let dir = tempdir_in(".").expect("tempdir");
        let config = write_cluster_config(
            dir.path(),
            r#"import json
fn main() ::
    return json.parse("{\"config_version\":1,\"backend\":\"cpu\",\"vocab_size\":8,\"hidden_size\":4,\"seq_len\":4,\"batch_size\":2,\"lr\":0.1,\"dataset_path\":\"data.txt\",\"checkpoint_dir\":\"ckpt\",\"max_steps\":2,\"save_every\":1,\"log_every\":1,\"tokenizer_train\":{\"path\":\"data.txt\",\"vocab_size\":8}}")
::
main()
"#,
        );
        let manifest = crate::systems::build_cluster_command_manifest(&[
            "run".to_string(),
            "--dry-run".to_string(),
            "--json".to_string(),
            "--output".to_string(),
            "artifacts/cluster/run.json".to_string(),
            config.to_string_lossy().to_string(),
        ])
        .expect("parse");
        let parsed = crate::cluster_runtime::cluster_args_from_manifest(&manifest);
        assert!(parsed.dry_run);
        assert!(parsed.json);
        assert_eq!(
            parsed.output,
            Some(PathBuf::from("artifacts/cluster/run.json"))
        );
        assert_eq!(parsed.config_path, config);
    }

    #[test]
    fn parse_usize_csv_accepts_csv() {
        let values =
            crate::cluster_runtime::parse_usize_csv("0, 1,2", 3, "dist.device_map").expect("csv");
        assert_eq!(values, vec![0, 1, 2]);
    }

    #[test]
    fn parse_usize_csv_rejects_invalid_values() {
        let err = crate::cluster_runtime::parse_usize_csv("0,nope", 2, "dist.device_map")
            .expect_err("invalid");
        assert!(err.contains("invalid value"));
    }

    #[test]
    fn build_cluster_plan_supports_multi_node_dist_contract() {
        let dir = tempdir_in(".").expect("tempdir");
        let config = write_cluster_config(
            dir.path(),
            r#"import json
fn main() ::
    return json.parse("{\"config_version\":1,\"backend\":\"cpu\",\"vocab_size\":8,\"hidden_size\":4,\"seq_len\":4,\"batch_size\":2,\"lr\":0.1,\"dataset_path\":\"data.txt\",\"checkpoint_dir\":\"ckpt\",\"max_steps\":2,\"save_every\":1,\"log_every\":1,\"tokenizer_train\":{\"path\":\"data.txt\",\"vocab_size\":8},\"world_size\":2,\"rank\":0,\"dist\":{\"topology\":\"multi-node\",\"rendezvous\":\"tcp://127.0.0.1:29500\",\"retry_budget\":2,\"device_map\":[0,1],\"hosts\":[\"node-a\",\"node-b\"],\"host_map\":[0,1]}}")
::
main()
"#,
        );
        let plan = crate::cluster_runtime::build_cluster_plan(&config).expect("plan");
        assert_eq!(plan.topology, "multi-node");
        assert_eq!(plan.hosts, vec!["node-a".to_string(), "node-b".to_string()]);
        assert_eq!(plan.host_map, vec![0, 1]);
        assert_eq!(plan.rank_plans[1].host, "node-b");
        assert_eq!(
            plan.rank_plans[1]
                .env
                .get("ENKAI_CLUSTER_HOST")
                .map(String::as_str),
            Some("node-b")
        );
    }

    #[test]
    fn build_cluster_plan_rejects_multi_node_env_rendezvous() {
        let dir = tempdir_in(".").expect("tempdir");
        let config = write_cluster_config(
            dir.path(),
            r#"import json
fn main() ::
    return json.parse("{\"config_version\":1,\"backend\":\"cpu\",\"vocab_size\":8,\"hidden_size\":4,\"seq_len\":4,\"batch_size\":2,\"lr\":0.1,\"dataset_path\":\"data.txt\",\"checkpoint_dir\":\"ckpt\",\"max_steps\":2,\"save_every\":1,\"log_every\":1,\"tokenizer_train\":{\"path\":\"data.txt\",\"vocab_size\":8},\"world_size\":2,\"rank\":0,\"dist\":{\"topology\":\"multi-node\",\"rendezvous\":\"env://\",\"retry_budget\":2,\"device_map\":[0,1],\"hosts\":[\"node-a\",\"node-b\"],\"host_map\":[0,1]}}")
::
main()
"#,
        );
        let err = crate::cluster_runtime::build_cluster_plan(&config).expect_err("env rendezvous");
        assert!(err.contains("tcp://<host>:<port>"));
    }

    #[test]
    fn build_cluster_plan_supports_simulation_world_partition_contract() {
        let dir = tempdir_in(".").expect("tempdir");
        let sim_target = dir.path().join("sim_world.enk");
        fs::write(&sim_target, "fn main() ::\n    return 0\n::\nmain()\n").expect("sim write");
        let config = write_cluster_config(
            dir.path(),
            r#"import json
fn main() ::
    return json.parse("{\"config_version\":1,\"backend\":\"cpu\",\"vocab_size\":8,\"hidden_size\":4,\"seq_len\":4,\"batch_size\":2,\"lr\":0.1,\"dataset_path\":\"data.txt\",\"checkpoint_dir\":\"ckpt\",\"max_steps\":2,\"save_every\":1,\"log_every\":1,\"tokenizer_train\":{\"path\":\"data.txt\",\"vocab_size\":8},\"world_size\":2,\"rank\":0,\"workload\":\"simulation\",\"dist\":{\"topology\":\"multi-node\",\"rendezvous\":\"tcp://127.0.0.1:29500\",\"retry_budget\":2,\"device_map\":[0,1],\"hosts\":[\"node-a\",\"node-b\"],\"host_map\":[0,1]},\"simulation\":{\"target\":\"sim_world.enk\",\"partition_count\":2,\"total_steps\":6,\"step_window\":2,\"snapshot_interval\":2,\"route_policy\":\"deterministic-ring\",\"recovery_dir\":\"recovery\",\"seed\":11}}")
::
main()
"#,
        );
        let plan = crate::cluster_runtime::build_cluster_plan(&config).expect("plan");
        let simulation = plan.simulation.expect("simulation");
        assert_eq!(simulation.partition_count, 2);
        assert_eq!(simulation.total_steps, 6);
        assert_eq!(simulation.step_window, 2);
        assert_eq!(simulation.route_policy, "deterministic-ring");
        assert!(simulation.target.ends_with("sim_world.enk"));
        assert_eq!(
            plan.rank_plans[1]
                .env
                .get("ENKAI_SIM_ROUTE_NEXT_RANK")
                .map(String::as_str),
            Some("0")
        );
    }
}
