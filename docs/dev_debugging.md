# Developer Debugging

Use these flags when diagnosing compiler or VM behavior.

```powershell
enkai run --disasm file.enk
enkai run --trace-vm file.enk
enkai run --trace-task file.enk
enkai run --trace-net file.enk
```

`--disasm` prints bytecode before execution. `--trace-vm` prints VM instruction
execution and stack state. Task and network tracing are useful for runtime and
transport debugging.

For normal user programs, prefer this loop first:

```powershell
enkai fmt --check file.enk
enkai check file.enk
enkai run file.enk
```
