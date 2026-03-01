Enkai Roadmap

Note:
- Historical milestones below capture the path that led to current releases.
- Current production release line is v1.4.0.
- Use `docs/Enkai.spec` as the source of truth for current language behavior.

v0.1 (done)
- Lexer/parser/AST with :: blocks
- Tree-walk interpreter
- Minimal types + control flow
- Minimal stdlib stubs
- CLI run/fmt/test stubs

v0.2 (done)
- Modules + use resolution
- Enkai run . with Enkai.toml + src/main.enk
- Better diagnostics (line/col + snippet)
- Runtime stack traces
- Minimal formatter + validation
- Policy enforcement MVP (default deny + allow rules)

v0.3 (done)
- Module exports/import rules (pub/private, re-export)
- Policy filters enforcement (domains, path_prefix)
- Diagnostics with labeled spans
- Local path dependencies in Enkai.toml
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


