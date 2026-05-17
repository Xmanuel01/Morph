# Tensor FFI Opaque Handles

`v4.0.0` closes the bounded tensor FFI handle-hardening tranche. The tensor runtime still transports handles through C-compatible `i64` slots for ABI compatibility, but those values are not raw object identities or sequential IDs. They are opaque capability tokens.

## Handle Properties

Production tensor handles now require:

- a registry-kind tag for tensor, device, optimizer, scaler, and LM session handles;
- a per-process checksum derived from a process secret and handle payload;
- registry membership before object access;
- stale-handle rejection after free;
- wrong-kind rejection before registry lookup;
- non-sequential token generation.

A caller may store and pass a handle token back to the FFI API, but must not inspect, forge, increment, or otherwise derive new handles from it.

## Failure Modes

The runtime reports deterministic errors for:

- untagged integer values;
- forged handles with invalid checksums;
- wrong-kind handles, such as using a tensor handle where a device handle is required;
- stale handles after free;
- unknown registry entries.

## Policy Export

The machine-readable policy is exported through:

```text
enkai_tensor_handle_abi_policy(out_json, out_len)
```

The JSON payload records that raw sequential IDs are not allowed and that kind, checksum, registry membership, stale, and wrong-kind checks are required.

## Verification

Run:

```powershell
$torchLib=(py -3.11 -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")
$env:LIBTORCH_USE_PYTORCH='1'
$env:PATH='C:\Users\USER\AppData\Local\Programs\Python\Python311;C:\Users\USER\AppData\Local\Programs\Python\Python311\Scripts;' + $torchLib + ';' + $env:PATH
$env:CARGO_BUILD_JOBS='1'
$env:RUSTFLAGS='-C debuginfo=0'
py scripts\verify_v4_0_0_tensor_ffi_opaque_handles.py --workspace . --run-tests
```

The verifier checks source markers, dedicated handle-hardening tests, AMP/scaler handle coverage, and writes:

```text
artifacts/readiness/v4_0_0_tensor_ffi_opaque_handles.json
```

This proof closes the bounded FFI opaque-handle requirement for the tensor ABI. It does not claim that every downstream consumer has migrated away from the legacy integer transport slot; it claims those slots no longer expose raw object identity and are hardened opaque capability tokens.
