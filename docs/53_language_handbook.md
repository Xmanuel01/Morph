# Enkai Language Handbook

This handbook is the learner-oriented overview for writing Enkai programs. It
summarizes the syntax, CLI, standard library, safety model, AI-native APIs, and
current production-proof boundaries in one place.

For the normative reference, read `docs/Enkai.spec`. For release status and
proof artifacts, read `docs/roadmap.md` and `docs/48_release_dashboard.md`.

## 1. Official CLI

Use lowercase `enkai` commands:

```text
enkai run <file.enkai>      Execute a file or supported project entry.
enkai check <file.enkai>    Check syntax, types, imports, and policy rules.
enkai fmt <file.enkai>      Format source using the official style.
enkai build [dir]           Build/check a project and refresh build metadata.
enkai test [dir]            Run project tests.
enkai train <config.enk>    Run a training config.
enkai pretrain <config.enk> Run a pretraining config.
enkai eval <config.enk>     Evaluate using a config/checkpoint.
enkai serve [options]       Start a serving entrypoint.
enkai help                  Show available commands.
enkai version               Print the version.
```

`enkai safari` is reserved for a future interactive workspace. It is not the
normal way to run files.

## 2. First Program

```enkai
import std::io

policy default ::
    allow io.write
::policy

fn main() ::
    let _ := io.stdout_write_text("Hello from Enkai\n")
::fn
```

Run it:

```powershell
enkai run hello.enkai
```

Check it without running side effects:

```powershell
enkai check hello.enkai
```

## 3. Block Syntax

Blocks open with `::`. Plain closers remain valid:

```enkai
if ready ::
    line("ready")
::
```

Tagged closers are preferred for committed code, nested blocks, functions,
policies, and loops:

```enkai
fn clamp01(value: Float) -> Float ::
    if value < 0.0 ::
        return 0.0
    ::if

    if value > 1.0 ::
        return 1.0
    ::if

    return value
::fn
```

Supported tagged closers include `::fn`, `::if`, `::else`, `::while`,
`::for`, `::policy`, `::match`, `::struct`, `::enum`, `::impl`, and
`::module`.

## 4. Variables, Mutation, And Constants

`let` is immutable:

```enkai
let name := "Nairobi"
let count: Int := 3
```

`mut` is required for reassignment:

```enkai
mut index := 0
while index < 3 ::
    index := index + 1
::while
```

`const` is for compile-time constants:

```enkai
const COUNTRY := "Kenya"
const MAX_BATCH := 128
```

Reassigning `let` or `const` is an error. Function parameters are immutable by
default; copy into a local `mut` binding if a function needs a changing value.

## 5. Imports

Everything outside the core language must be imported explicitly:

```enkai
import std::io
import std::json
import std::tensor
```

After `import std::json`, use `json.enkai(value)` for JSON encoding. The older
`json.stringify(value)` spelling remains a compatibility alias, but new code
should prefer `json.enkai(value)`.

Missing imports are reported by `enkai check` where possible:

```text
ImportError: `json.enkai` requires `import std::json`
```

## 6. Policies

Policy blocks make side effects explicit:

```enkai
import std::io

policy default ::
    allow io.write
    deny net.request
::policy
```

Known standard-library effects are checked before runtime where possible.
Explicit `deny` rules override matching `allow` rules.

Common policy capabilities include:

```text
io.write
fs.read
fs.write
env.read
env.write
process.spawn
process.control
db.read
db.write
net.tls
net.serve
net.http
io.log
time.sleep
```

## 7. Types

Core types:

```text
Int
Float
Bool
String
Void
Any
```

Collections and AI-native types:

```enkai
let names: Array[String] := ["Nairobi", "Mombasa"]
let dense: Vector[Float] := [0.1, 0.2, 0.3]
let sparse_values: SparseVector[Float] := sparse.vector()
let matrix: Tensor[Float, 2] := tensor.zeros([3, 3], "f32", 0)
```

Array inference rules:

- `["a", "b"]` infers `Array[String]`.
- `[1, 2, 3]` infers `Array[Int]`.
- `[1, 2.5]` infers `Array[Float]`.
- `[]` requires an explicit type.
- mixed unrelated element types require an explicit type such as `Array[Any]`.

## 8. Functions

```enkai
fn add(a: Int, b: Int) -> Int ::
    return a + b
::fn
```

If no return type is written, the function returns `Void`.

```enkai
fn line(text: String) ::
    let _ := io.stdout_write_text(text)
    let _n := io.stdout_write_text("\n")
::fn
```

## 9. Modules And Visibility

Standard modules use `std::name`:

```enkai
import std::json
import std::tensor
```

Local modules use project paths:

```enkai
import app::math as math
```

Export only stable API symbols:

```enkai
export fn add(a: Int, b: Int) -> Int ::
    return a + b
::fn
```

Private helpers do not use `export`.

## 10. Useful Standard Modules

General modules:

```text
std::io              stdin/stdout/stderr and text/byte file helpers
std::json            JSON parse and encode
std::array           array length/type helpers
std::vector          dense vector helpers
std::sparse          sparse vectors and matrices
std::tensor          tensor and autodiff API
std::nn              neural-network layer helpers
std::optim           optimizer helpers
std::data            tokenizer and dataset pipeline helpers
std::checkpoint      checkpoint manifest helpers
std::model           model API helpers
std::env             environment variables and cwd
std::fsx             native-backed file byte IO
std::hash            SHA-256 helpers
std::http            HTTP routing, middleware, client/server helpers
std::db              database helpers
std::process         process helpers
std::time            time helpers
std::tls             TLS inspection helpers
std::zstd            compression helpers
```

Bounded runtime guardrails:

```text
ENKAI_SPARSE_MAX_INDEX       maximum sparse row/column/vector index
ENKAI_SPARSE_MAX_NNZ         maximum sparse non-zero entries per value
ENKAI_DENSE_MAX_LEN          maximum dense list/buffer length accepted by sparse ops
ENKAI_EVENT_MAX_LEN          maximum pending events per event queue
ENKAI_POOL_MAX_CAPACITY      maximum fixed/growable pool capacity
ENKAI_SPATIAL_MAX_ENTITIES   maximum entities in one spatial index
ENKAI_SNN_MAX_NEURONS        maximum neurons in one SNN network
ENKAI_SNN_MAX_BATCH          maximum rows accepted by snn.step_batch
```

These defaults protect `std::sparse`, `std::event`, `std::pool`, and `std::spatial`
from accidental unbounded allocation while keeping deterministic VM fallback behavior.
`std::spatial` uses a packed R-tree index for radius, nearest-neighbor, and occupancy
queries, with stable tie-breaking by entity id.
`std::snn` exposes `snn.step` for one input vector and `snn.step_batch` for row-major
batches. Batched stepping uses the same deterministic kernel as scalar stepping and uses
the native runtime kernel when `enkai_native` acceleration is available.

## 11. JSON Example

```enkai
import std::io
import std::json

policy default ::
    allow io.write
::policy

fn main() ::
    let payload := json.enkai({"county": "Nairobi", "score": 0.92})
    let _ := io.stdout_write_text(payload + "\n")
::fn
```

## 12. Tensor Example

```enkai
import std::tensor

fn main() ::
    let cpu := tensor.device("cpu")
    let x := tensor.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])
    let y := tensor.matmul(x, x)
    let probs := tensor.softmax(y, 1)
    let shape := tensor.shape(probs)
::fn
```

The first-party CPU tensor path is deterministic and suitable for bounded
proofs. GPU production claims require hardware-backed CUDA/PyTorch evidence.

## 13. Training Config Example

Training, pretraining, and eval use config files that return a config record.
JSON is the most portable way to construct the record:

```enkai
import std::json

fn main() ::
    return json.parse("{\"config_version\":1,\"backend\":\"cpu\",\"vocab_size\":8,\"hidden_size\":4,\"seq_len\":4,\"batch_size\":2,\"lr\":0.1,\"dataset_path\":\"data.txt\",\"checkpoint_dir\":\"ckpt\",\"max_steps\":2,\"save_every\":1,\"log_every\":1,\"tokenizer_train\":{\"path\":\"data.txt\",\"vocab_size\":8}}")
::fn
```

Run:

```powershell
enkai train config.enk
enkai eval config.enk
```

## 14. Debugging Flow

Use this loop while learning:

```powershell
enkai fmt --check main.enkai
enkai check main.enkai
enkai run main.enkai
```

For VM-level debugging:

```powershell
enkai run --trace-vm main.enkai
enkai run --disasm main.enkai
```

## 15. Current Production Boundaries

Enkai has closed shipped strict-selfhost proof lines and active global
self-host/GPU proof work. Do not treat proof-ready source as hardware sign-off.

Current CUDA-first LLM runtime claims require:

- real CUDA hardware preflight,
- PyTorch CUDA reference availability,
- archived correctness/throughput/memory/checkpoint evidence,
- green verifier artifacts,
- no failed distributed proof if distributed claims are made.

If those artifacts are missing or red, the correct status is proof-ready or
in-progress, not production-closed.
