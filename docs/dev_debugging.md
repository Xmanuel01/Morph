# Developer Debugging Guide

- `Enkai run --disasm file.enk` prints bytecode before execution.
- `Enkai run --trace-vm file.enk` prints each instruction and current stack.
- Reading disassembly: instruction index, source line, opcode.
- Adding an opcode:
  1) Add to `enkaic/src/bytecode.rs`.
  2) Emit it in the compiler.
  3) Execute it in `enkairt/src/vm.rs` with stack safety.
  4) Add tests.
- Stack issues:
  - Underflow: ensure you push before reading; trace-vm will show empty stack.
  - Overflow: keep stack usage bounded by control flow; pops should mirror pushes.


