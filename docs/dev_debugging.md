# Developer Debugging Guide

- `morph run --disasm file.morph` prints bytecode before execution.
- `morph run --trace-vm file.morph` prints each instruction and current stack.
- Reading disassembly: instruction index, source line, opcode.
- Adding an opcode:
  1) Add to `morphc/src/bytecode.rs`.
  2) Emit it in the compiler.
  3) Execute it in `morphrt/src/vm.rs` with stack safety.
  4) Add tests.
- Stack issues:
  - Underflow: ensure you push before reading; trace-vm will show empty stack.
  - Overflow: keep stack usage bounded by control flow; pops should mirror pushes.
