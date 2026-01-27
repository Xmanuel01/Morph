# Globals

- Top-level `let` defines a global variable.
- VM opcodes: `DefineGlobal`, `LoadGlobal`, `StoreGlobal`.
- Globals persist for program lifetime and are visible to functions.
- Type checker enforces assignments against the declared/global type.

