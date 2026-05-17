# Native SNN Batched Kernels

This document defines the bounded production claim for Enkai's native SNN runtime closure.

## Closed Scope

The closed surface is the first-party native batched SNN step path used by `std::snn.step_batch`.

The claim is intentionally specific:

- `std::snn.step_batch(network, inputs)` is available as the public Enkai API.
- The VM routes the batch through one native `sim_snn_step_batch` call when native acceleration is available.
- The native kernel updates the whole batch deterministically using row-major dense `Float` inputs.
- The recurrent step uses an adjacency list frontier, not a full synapse scan per neuron.
- Empty, malformed, non-finite, or invalid-handle inputs fail closed.
- VM fallback and native execution are tested for deterministic equivalence on seeded scenarios.
- The closure has a dedicated contract, verifier, and readiness artifact.

## Not Claimed

This is not a claim that Enkai has a full neuromorphic hardware backend, GPU SNN runtime, or research-grade spiking neural network ecosystem. Those are separate future surfaces.

## Public API

```enkai
import std::snn

let net := snn.make(3)
snn.set_threshold(net, 0, 0.4)
snn.connect(net, 0, 1, 0.5)
let spikes := snn.step_batch(net, [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
```

`spikes` is an array of fired-neuron index arrays, one row per input row.

## Native ABI

The native ABI entrypoint is:

```text
sim_snn_step_batch(handle, row_major_f64_input_bytes) -> potentials_f64_then_batch_spikes_u8
```

The payload layout is deterministic:

- First `neuron_count * 8` bytes are final membrane potentials as little-endian `f64`.
- Remaining bytes are `batch_len * neuron_count` spike bits as `u8` values.

The native policy export `sim_snn_runtime_policy_json` records this contract for verifier consumption.

## Verification

Run the dedicated verifier:

```powershell
py scripts\verify_v4_0_0_native_snn_batched_kernels.py --workspace . --run-tests
```

The verifier requires:

- Native batch kernel export.
- Public `std::snn.step_batch` export.
- One native call per VM batch path.
- Fail-closed invalid input tests.
- Native/VM deterministic equivalence tests.
- Machine-readable native runtime policy export.

The proof artifact is:

```text
artifacts/readiness/v4_0_0_native_snn_batched_kernels.json
```
