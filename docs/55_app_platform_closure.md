# App Platform Closure

`v4.0.0` adds a bounded app-platform closure tranche for the MySQL, gRPC, and
mobile surfaces. The goal is to replace scaffold-only claims with explicit
contracts, fail-closed behavior, and reproducible evidence.

## Scope

The closure covers:

- MySQL standard-library API parity for open, close, exec, batch exec, query,
  and transaction control.
- MySQL fail-closed behavior when no live database service or invalid handle is
  available.
- Static policy mapping for MySQL read/write operations.
- Local gRPC end-to-end readiness, ready/health probes, chat streaming, and
  conversation persistence evidence.
- Mobile strict deploy validation, SDK snapshot generation, and package/app
  manifest evidence.

The closure does not claim:

- A live external MySQL production service has been provisioned and audited.
- A mobile app has been store-submitted or signed for production distribution.
- A managed cloud gRPC deployment has been load-tested.

Those claims require separate live-service or deployment evidence.

## MySQL API

Use `std::db` for the bounded MySQL surface:

```enkai
import std::db

policy default ::
    allow db.read
    allow db.write
::policy

let handle := db.mysql_open("mysql://user:pass@localhost:3306/app")
let started := db.mysql_transaction_begin(handle)
let rows_changed := db.mysql_exec_many(handle, "insert into events(id) values (1)", 1)
let committed := db.mysql_transaction_commit(handle)
let rows := db.mysql_query(handle, "select id from events")
let closed := db.mysql_close(handle)
```

Transaction helpers return `false` on invalid handles. `mysql_exec_many` returns
`-1` on invalid handles. This is intentional fail-closed behavior.

## Policy

MySQL operations are policy-aware:

- `mysql_query` is a `db.read` operation.
- `mysql_open`, `mysql_close`, `mysql_exec`, `mysql_exec_many`, and transaction
  helpers are `db.write` operations.

Missing policy permission must fail before silently executing a database action.

## Verification

Run the closure verifier from the repository root:

```powershell
py scripts\verify_v4_0_0_app_platform_closure.py --workspace . --run
```

The verifier checks:

- `enkai/contracts/v4_0_0_app_platform_closure.json`
- `artifacts/readiness/grpc_smoke.json`
- `artifacts/readiness/grpc_evidence_verify.json`
- `artifacts/readiness/deploy_mobile_smoke.json`
- `artifacts/readiness/deploy_mobile_evidence_verify.json`
- `artifacts/readiness/v4_0_0_app_platform_closure.json`

Targeted tests:

```powershell
cargo test -p enkai_native mysql_transaction_apis_fail_closed_for_invalid_handles
cargo test -p enkairt --test ffi_modules std_db_mysql_transaction_apis_fail_closed_for_invalid_handle
```

## Production Claim Rule

For this tranche, "production-grade" means contract-verified bounded behavior:
API surface, policy mapping, fail-closed invalid-handle behavior, local gRPC
end-to-end evidence, and mobile package validation evidence are green.

It does not mean external service production closure. Live MySQL, managed gRPC,
and signed mobile release proofs must be added as separate hardware/service
tranches before making those broader claims.
