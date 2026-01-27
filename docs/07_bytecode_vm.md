# Bytecode & VM (v0.4)

Pipeline: source → AST → type-check → bytecode → VM.

Chunk: instructions + constant pool + line map.

Core opcodes:
- Stack/constants: `Const`, `Pop`
- Globals: `DefineGlobal`, `LoadGlobal`, `StoreGlobal`
- Locals: `LoadLocal`, `StoreLocal`
- Arith: `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Neg`
- Logic: `Not`
- Compare: `Eq`, `Neq`, `Lt`, `Gt`, `Le`, `Ge`
- Control: `Jump`, `JumpIfFalse`
- Calls: `Call`, `Return`

Short-circuit is compiled with `JumpIfFalse`/`Jump` sequences, not dedicated opcodes.

Debugging:
- `--disasm` prints bytecode
- `--trace-vm` prints each instruction + stack

