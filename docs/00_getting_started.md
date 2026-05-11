# Getting Started With Enkai

This guide is the fastest path from installing Enkai to writing useful programs.
It focuses on the language surface a learner needs first: files, commands,
syntax, imports, types, policies, and common runtime modules.

## 1. Check The CLI

After installing Enkai, confirm the executable is on your `PATH`:

```powershell
enkai --version
enkai help
```

The official execution command is:

```powershell
enkai run main.enkai
```

Use these commands while learning:

```text
enkai run <file.enkai>      Run an Enkai source file or project directory.
enkai check <file.enkai>    Check syntax, types, imports, and policies.
enkai fmt <file.enkai>      Format source using the official style.
enkai build [dir]           Build/check a project without running it.
enkai test [dir]            Run project tests.
enkai help                  Show the full CLI surface.
enkai version               Print the compiler/runtime version.
```

`enkai safari` is reserved for a future interactive workspace and is not the
normal run command.

## 2. Your First Program

Create `hello.enkai`:

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

Check it without running:

```powershell
enkai check hello.enkai
```

## 3. The Core Shape Of Enkai

Enkai is designed around a small set of readable rules:

- blocks open with `::`
- plain `::` can close any block
- tagged closers such as `::fn`, `::if`, `::while`, and `::policy` are preferred for larger code
- `let` is immutable
- `mut` is mutable
- `const` is a compile-time constant
- modules are imported explicitly, for example `import std::json`
- policy blocks declare permissions before side effects run

Example:

```enkai
import std::io
import std::json

policy default ::
    allow io.write
::policy

fn line(text: String) ::
    let _ := io.stdout_write_text(text)
    let _n := io.stdout_write_text("\n")
::fn

fn main() ::
    const COUNTRY := "Kenya"
    let counties: Array[String] := ["Nairobi", "Mombasa", "Kisumu"]
    mut index := 0

    while index < counties.length ::
        let payload := json.enkai(counties[index])
        line(COUNTRY + ": " + payload)
        index := index + 1
    ::while
::fn
```

## 4. Files And Extensions

Enkai source files commonly use:

```text
.enk
.enkai
```

Use either consistently in your project. Most examples in the docs use `.enkai`
for entry files and `.enk` for compact examples or bootstrap/runtime sources.

## 5. Projects

A simple project can be a directory with one entry file:

```text
my_project/
  main.enkai
```

Run the entry file directly:

```powershell
enkai run my_project\main.enkai
```

Or run/check a directory when the project metadata supports it:

```powershell
enkai run my_project
enkai check my_project
```

## 6. What To Read Next

Read these docs in order:

1. `docs/53_language_handbook.md` for the complete learner overview.
2. `docs/01_syntax.md` for blocks, variables, imports, policies, and examples.
3. `docs/02_types.md` for primitives, arrays, vectors, tensors, and inference.
4. `docs/03_functions.md` for functions, returns, and parameters.
5. `docs/09_modules.md` for imports and multi-file programs.
6. `docs/52_cli_and_style.md` for official CLI and formatting style.
7. `docs/tensor_api.md` for AI-native tensor APIs and current production boundaries.
8. `docs/Enkai.spec` for the normative language reference.
