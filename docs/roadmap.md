Morph Roadmap

v0.1 (done)
- Lexer/parser/AST with :: blocks
- Tree-walk interpreter
- Minimal types + control flow
- Minimal stdlib stubs
- CLI run/fmt/test stubs

v0.2 (done)
- Modules + use resolution
- morph run . with morph.toml + src/main.morph
- Better diagnostics (line/col + snippet)
- Runtime stack traces
- Minimal formatter + validation
- Policy enforcement MVP (default deny + allow rules)

v0.3 (done)
- Module exports/import rules (pub/private, re-export)
- Policy filters enforcement (domains, path_prefix)
- Diagnostics with labeled spans
- Local path dependencies in morph.toml
- Expand stdlib: strings + fs (policy-gated)
- Keep AI primitives as stubs unless testable

v0.5 (planned)
- LSP + formatter improvements
- Async runtime (spawn/await)
- HTTP server library
- Agent + tool abstractions

v1.0 (planned)
- Fast compiler backend
- Strong sandboxing + capability permissions
- Stable stdlib
- Polished docs + examples
