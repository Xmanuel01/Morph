#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the v3.3.0 self-host systems control-plane slice."
    )
    parser.add_argument("--enkai-bin", required=True, help="Path to enkai executable")
    parser.add_argument(
        "--contract",
        default="enkai/contracts/selfhost_systems_controlplane_v3_3_0.json",
        help="Path to the control-plane contract JSON",
    )
    parser.add_argument(
        "--output",
        default="artifacts/readiness/strict_selfhost_systems_controlplane_slice.json",
        help="Path to write the verification report JSON",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> dict:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {
        "command": command,
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def ensure(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    values: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        values.append(json.loads(line))
    return values


def write_cluster_config(path: Path) -> None:
    path.write_text(
        """import json
fn main() ::
    return json.parse("{\\"config_version\\":1,\\"backend\\":\\"cpu\\",\\"vocab_size\\":8,\\"hidden_size\\":4,\\"seq_len\\":4,\\"batch_size\\":2,\\"lr\\":0.1,\\"dataset_path\\":\\"data.txt\\",\\"checkpoint_dir\\":\\"ckpt\\",\\"max_steps\\":2,\\"save_every\\":1,\\"log_every\\":1,\\"tokenizer_train\\":{\\"path\\":\\"data.txt\\",\\"vocab_size\\":8},\\"world_size\\":2,\\"rank\\":0,\\"dist\\":{\\"topology\\":\\"multi-node\\",\\"rendezvous\\":\\"tcp://127.0.0.1:29500\\",\\"retry_budget\\":2,\\"device_map\\":[0,1],\\"hosts\\":[\\"node-a\\",\\"node-b\\"],\\"host_map\\":[0,1]}}")
::
main()
""",
        encoding="utf-8",
    )


def write_cluster_sim_target(path: Path) -> None:
    path.write_text(
        """import std::sim
fn main() ::
    let w := sim.make_seeded(8, 7)
    sim.schedule(w, 1.0, 9)
    sim.run(w, 1)
    return sim.snapshot(w)
::
main()
""",
        encoding="utf-8",
    )


def create_backend_project(root: Path) -> None:
    contents = {
        root / "enkai.toml": 'name = "backend-project"\nversion = "0.1.0"\n',
        root / "src" / "main.enk": "fn main() ::\n    return 0\n::\nmain()\n",
        root / "contracts" / "backend_api.snapshot.json": '{"api":"backend"}\n',
        root / "contracts" / "conversation_state.schema.json": '{"title":"conversation_state"}\n',
        root / "contracts" / "grpc_api.snapshot.json": '{\n  "package": "enkai.chat.v1",\n  "services": [\n    {\n      "name": "ChatService",\n      "methods": [\n        {"name": "Chat"},\n        {"name": "StreamChat"}\n      ]\n    }\n  ]\n}\n',
        root / "contracts" / "worker_queue.snapshot.json": '{\n  "queue_kind": "file_jsonl",\n  "durable_enqueue": true,\n  "dead_letter": true\n}\n',
        root / "contracts" / "db_engines.snapshot.json": '{\n  "engines": ["sqlite","postgres","mysql"],\n  "tables": ["schema_migrations"]\n}\n',
        root / "contracts" / "enkai_chat.proto": 'syntax = "proto3";\npackage enkai.chat.v1;\nservice ChatService {\n  rpc Chat (ChatRequest) returns (ChatResponse);\n  rpc StreamChat (ChatRequest) returns (stream ChatResponse);\n}\nmessage ChatRequest {}\nmessage ChatResponse {}\n',
        root / "contracts" / "deploy_env.snapshot.json": '{\n  "profile": "backend",\n  "required_env": ["ENKAI_API_KEY", "ENKAI_DB_URL"]\n}\n',
        root / "scripts" / "validate_env_contract.py": 'import sys\nsys.exit(9)\n',
        root / ".env.example": "ENKAI_APP_PROFILE=backend\nENKAI_API_KEY=test-key\nENKAI_DB_URL=postgres://localhost/enkai\nENKAI_SERVE_PORT=8080\nENKAI_GRPC_PORT=9090\nENKAI_DB_POOL_MAX=4\nENKAI_DB_ENGINE=postgres\n",
        root / "migrations" / "001_conversation_state.sql": 'CREATE TABLE IF NOT EXISTS schema_migrations(id INTEGER PRIMARY KEY);\nCREATE TABLE IF NOT EXISTS conversation_events(updated_ms INTEGER);\n',
        root / "migrations" / "002_conversation_state_index.sql": 'CREATE INDEX IF NOT EXISTS idx_conversation_events_updated_ms ON conversation_events(updated_ms);\n',
        root / "worker" / "handler.enk": "fn main() ::\n    return 0\n::\nmain()\n",
        root / "deploy" / "docker" / "Dockerfile": "FROM scratch\n",
        root / "deploy" / "docker-compose.yml": "services:\n  backend:\n    environment:\n      ENKAI_API_KEY: ${ENKAI_API_KEY}\n      ENKAI_DB_URL: ${ENKAI_DB_URL}\n",
        root / "deploy" / "systemd" / "enkai-worker.service": "[Service]\nEnvironmentFile=/opt/enkai-app/.env\nExecStart=/usr/local/bin/enkai worker run --queue default --handler worker/handler.enk\nRestart=on-failure\n",
        root / "deploy" / "systemd" / "enkai-backend.service": "[Service]\nEnvironmentFile=/opt/enkai-app/.env\nExecStart=/usr/local/bin/enkai serve --host 0.0.0.0 --port 8080 .\nEnvironment=ENKAI_PROFILE=backend\nRestart=on-failure\n",
    }
    for file_path, text in contents.items():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    enkai_bin = Path(args.enkai_bin).resolve()
    contract_path = (repo_root / args.contract).resolve()
    output_path = (repo_root / args.output).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(
        prefix="enkai_systems_controlplane_", dir=str(repo_root)
    ) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        worker_state = temp_dir / "worker_state"
        worker_state.mkdir(parents=True, exist_ok=True)
        handler = temp_dir / "handler.enk"
        handler.write_text("fn main() ::\n    return 0\n::\nmain()\n", encoding="utf-8")
        backend_project = temp_dir / "backend_project"
        create_backend_project(backend_project)
        cluster_config = temp_dir / "cluster_config.enk"
        write_cluster_config(cluster_config)
        cluster_sim_target = temp_dir / "cluster_sim_world.enk"
        write_cluster_sim_target(cluster_sim_target)
        cluster_run_config = temp_dir / "cluster_run_config.enk"
        cluster_run_config.write_text(
            """import json
fn main() ::
    return json.parse("{\\"config_version\\":1,\\"backend\\":\\"cpu\\",\\"vocab_size\\":8,\\"hidden_size\\":4,\\"seq_len\\":4,\\"batch_size\\":2,\\"lr\\":0.1,\\"dataset_path\\":\\"data.txt\\",\\"checkpoint_dir\\":\\"ckpt\\",\\"max_steps\\":2,\\"save_every\\":1,\\"log_every\\":1,\\"tokenizer_train\\":{\\"path\\":\\"data.txt\\",\\"vocab_size\\":8},\\"world_size\\":1,\\"rank\\":0,\\"workload\\":\\"simulation\\",\\"dist\\":{\\"topology\\":\\"multi-node\\",\\"rendezvous\\":\\"tcp://127.0.0.1:29500\\",\\"retry_budget\\":1,\\"device_map\\":[0],\\"hosts\\":[\\"node-a\\"],\\"host_map\\":[0]},\\"simulation\\":{\\"target\\":\\"cluster_sim_world.enk\\",\\"partition_count\\":1,\\"total_steps\\":2,\\"step_window\\":1,\\"snapshot_interval\\":1,\\"route_policy\\":\\"deterministic-ring\\",\\"recovery_dir\\":\\"recovery\\",\\"seed\\":11}}")
::
main()
""",
            encoding="utf-8",
        )

        enqueue_manifest = temp_dir / "worker_enqueue.json"
        enqueue_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "worker-manifest",
                "enqueue",
                "--queue",
                "jobs",
                "--dir",
                str(worker_state),
                "--payload",
                "{\"job\":1}",
                "--max-attempts",
                "4",
                "--json",
                "--output",
                str(enqueue_manifest),
            ],
            repo_root,
        )

        run_manifest = temp_dir / "worker_run.json"
        run_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "worker-manifest",
                "run",
                "--queue",
                "jobs",
                "--dir",
                str(worker_state),
                "--handler",
                str(handler),
                "--once",
                "--json",
                "--output",
                str(run_manifest),
            ],
            repo_root,
        )

        worker_exec_output = temp_dir / "worker_exec_report.json"
        worker_exec_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "worker-exec",
                "--manifest",
                str(enqueue_manifest),
            ],
            repo_root,
        )
        worker_run_exec_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "worker-exec",
                "--manifest",
                str(run_manifest),
            ],
            repo_root,
        )
        queue_dir = worker_state / "queues" / "jobs"
        pending_path = queue_dir / "pending.jsonl"
        inflight_path = queue_dir / "inflight.jsonl"
        schedule_path = queue_dir / "scheduled.jsonl"
        state_path = queue_dir / "queue_state.json"
        pending_payload = read_jsonl(pending_path)

        deploy_manifest = temp_dir / "deploy_validate.json"
        deploy_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "deploy-manifest",
                "validate",
                str(backend_project),
                "--profile",
                contract["deploy"]["required_profile"],
                "--strict",
                "--json",
                "--output",
                str(deploy_manifest),
            ],
            repo_root,
        )

        deploy_exec_output = temp_dir / "deploy_exec_report.json"
        deploy_exec_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "deploy-exec",
                "--manifest",
                str(deploy_manifest),
            ],
            repo_root,
        )

        cluster_manifest = temp_dir / "cluster_plan.json"
        cluster_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "cluster-manifest",
                contract["cluster"]["required_subcommand"],
                "--json",
                str(cluster_config),
                "--output",
                str(cluster_manifest),
            ],
            repo_root,
        )

        cluster_exec_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "cluster-exec",
                "--manifest",
                str(cluster_manifest),
            ],
            repo_root,
        )

        cluster_run_manifest = temp_dir / "cluster_run.json"
        cluster_run_manifest_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "cluster-manifest",
                "run",
                "--json",
                str(cluster_run_config),
                "--output",
                str(cluster_run_manifest),
            ],
            repo_root,
        )
        cluster_run_exec_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "cluster-exec",
                "--manifest",
                str(cluster_run_manifest),
            ],
            repo_root,
        )

        report = {
            "schema_version": 1,
            "profile": contract.get("profile"),
            "contract": str(contract_path),
            "contract_version": contract.get("contract_version"),
            "all_passed": False,
            "cases": {
                "worker_enqueue": enqueue_result,
                "worker_run": run_result,
                "worker_exec_enqueue": worker_exec_result,
                "worker_exec_run": worker_run_exec_result,
                "deploy_validate": deploy_result,
                "deploy_exec": deploy_exec_result,
                "cluster_plan": cluster_result,
                "cluster_exec": cluster_exec_result,
                "cluster_run_manifest": cluster_run_manifest_result,
                "cluster_run_exec": cluster_run_exec_result,
            },
            "validations": {},
        }

        failures: list[str] = []
        ensure(enqueue_result["ok"], "worker enqueue manifest command failed", failures)
        ensure(run_result["ok"], "worker run manifest command failed", failures)
        ensure(worker_exec_result["ok"], "worker enqueue exec command failed", failures)
        ensure(worker_run_exec_result["ok"], "worker run exec command failed", failures)
        ensure(deploy_result["ok"], "deploy manifest command failed", failures)
        ensure(deploy_exec_result["ok"], "deploy exec command failed", failures)
        ensure(cluster_result["ok"], "cluster manifest command failed", failures)
        ensure(cluster_exec_result["ok"], "cluster exec command failed", failures)
        ensure(cluster_run_manifest_result["ok"], "cluster run manifest command failed", failures)
        ensure(cluster_run_exec_result["ok"], "cluster run exec command failed", failures)

        if not failures:
            enqueue_payload = json.loads(enqueue_manifest.read_text(encoding="utf-8"))
            run_payload = json.loads(run_manifest.read_text(encoding="utf-8"))
            deploy_payload = json.loads(deploy_manifest.read_text(encoding="utf-8"))
            cluster_payload = json.loads(cluster_manifest.read_text(encoding="utf-8"))
            cluster_run_payload = json.loads(cluster_run_manifest.read_text(encoding="utf-8"))
            deploy_exec_stdout = deploy_exec_result["stdout"]
            cluster_exec_stdout = cluster_exec_result["stdout"]
            cluster_run_exec_stdout = cluster_run_exec_result["stdout"]

            report["validations"] = {
                "worker_enqueue": enqueue_payload,
                "worker_run": run_payload,
                "worker_pending_after_exec": pending_payload,
                "deploy_validate": deploy_payload,
                "cluster_plan": cluster_payload,
                "cluster_run_manifest": cluster_run_payload,
                "deploy_exec_stdout": deploy_exec_stdout,
                "cluster_exec_stdout": cluster_exec_stdout,
                "cluster_run_exec_stdout": cluster_run_exec_stdout,
            }

            ensure(
                enqueue_payload["mode"] in contract["worker"]["required_modes"],
                "worker enqueue manifest mode mismatch",
                failures,
            )
            ensure(
                run_payload["mode"] in contract["worker"]["required_modes"],
                "worker run manifest mode mismatch",
                failures,
            )
            for field in contract["worker"]["required_enqueue_fields"]:
                ensure(field in enqueue_payload, f"missing enqueue field {field}", failures)
            for field in contract["worker"]["required_run_fields"]:
                ensure(field in run_payload, f"missing run field {field}", failures)
            ensure(
                enqueue_payload["retry_policy"]["max_attempts"] == enqueue_payload["max_attempts"],
                "worker enqueue retry_policy.max_attempts did not match max_attempts",
                failures,
            )
            ensure(
                enqueue_payload["backend_kind"] == contract["worker"]["required_backend_kind"],
                "worker enqueue backend_kind mismatch",
                failures,
            )
            ensure(
                run_payload["backend_kind"] == contract["worker"]["required_backend_kind"],
                "worker run backend_kind mismatch",
                failures,
            )
            ensure(
                run_payload["run_policy"]["drain_mode"] == ("once" if run_payload["once"] else "until_idle"),
                "worker run_policy.drain_mode did not match once flag",
                failures,
            )
            ensure(
                run_payload["run_policy"]["max_messages"] == (1 if run_payload["once"] else None),
                "worker run_policy.max_messages did not match expected drain limit",
                failures,
            )
            ensure(
                len(pending_payload) == 0,
                "worker exec run should leave no pending payloads in the queue",
                failures,
            )
            ensure(inflight_path.is_file(), "worker backend did not create inflight.jsonl", failures)
            ensure(schedule_path.is_file(), "worker backend did not create scheduled.jsonl", failures)
            ensure(state_path.is_file(), "worker backend did not create queue_state.json", failures)
            ensure(
                "status=acked" in worker_run_exec_result["stdout"],
                "worker exec run did not acknowledge the queued message",
                failures,
            )

            evaluated_project = deploy_payload["evaluated_project"]
            for field in contract["deploy"]["required_evaluation_fields"]:
                ensure(field in evaluated_project, f"missing deploy evaluation field {field}", failures)
            ensure(
                evaluated_project["missing_required_paths"] == 0,
                "deploy evaluation should report no missing required paths",
                failures,
            )
            ensure(
                "[deploy-validate] ok" in deploy_exec_stdout,
                "deploy exec did not report successful validation",
                failures,
            )

            evaluated_plan = cluster_payload["evaluated_plan"]
            for field in contract["cluster"]["required_plan_fields"]:
                ensure(field in evaluated_plan, f"missing cluster plan field {field}", failures)
            ensure(evaluated_plan["topology"] == "multi-node", "cluster topology mismatch", failures)
            ensure(evaluated_plan["world_size"] == 2, "cluster world_size mismatch", failures)
            ensure(
                "topology=multi-node" in cluster_exec_stdout,
                "cluster exec did not emit the planned topology report",
                failures,
            )
            ensure(
                f"status={contract['cluster']['required_simulation_run_status']}" in cluster_run_exec_stdout,
                "cluster run execution did not finish with ok status",
                failures,
            )
            simulation_stdout_log = temp_dir / "recovery" / "rank0" / "window_0000.stdout.log"
            ensure(
                simulation_stdout_log.is_file(),
                "cluster run did not produce the expected simulation stdout log",
                failures,
            )
            if simulation_stdout_log.is_file():
                stdout_log_text = simulation_stdout_log.read_text(encoding="utf-8")
                ensure(
                    contract["cluster"]["required_simulation_stdout_fragment"] in stdout_log_text,
                    "cluster simulation stdout log did not prove in-process execution",
                    failures,
                )

        report["all_passed"] = not failures
        report["failures"] = failures
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "ok" if report["all_passed"] else "failed",
                    "output": str(output_path),
                    "all_passed": report["all_passed"],
                },
                separators=(",", ":"),
            )
        )
        return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
