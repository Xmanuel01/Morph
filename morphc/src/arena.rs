use bumpalo::Bump;

/// CompilerContext holds arenas used during parsing/compilation.
/// The arena lets us allocate short-lived compiler data without
/// burdening the global allocator. It can be expanded later to
/// host AST allocations directly.
#[derive(Default)]
pub struct CompilerArena {
    pub bump: Bump,
}

impl CompilerArena {
    pub fn new() -> Self {
        Self { bump: Bump::new() }
    }
}
