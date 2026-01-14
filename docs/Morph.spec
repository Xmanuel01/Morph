# Morph Language Specification (v1.0 draft)

Status: draft. This document defines the surface syntax and core runtime rules
for Morph v1.0. The language is block-structured using "::" and avoids curly
braces for blocks.

Design goals:
- Readable "::" blocks and minimal punctuation.
- Safe-by-default execution with explicit capabilities.
- Fast runtime and compiler-friendly syntax.
- AI-native primitives (tools, agents, memory, policies).

-------------------------------------------------------------------------------
1. Lexical structure
-------------------------------------------------------------------------------

Source:
- Files are UTF-8 text. Identifiers are ASCII in v1.0.
- Whitespace is not significant, except for block delimiters (see section 2).
- Newlines separate statements unless inside (), [].

Comments:
- Line comments: // until end of line.
- Block comments: /* ... */ (nesting not allowed).

Identifiers:
- Regex: [A-Za-z_][A-Za-z0-9_]*
- Case-sensitive.

Keywords:
agent, allow, and, as, async, await, break, catch, continue, deny, else, enum,
false, fn, for, if, impl, in, let, match, memory, model, none, not, or, policy,
prompt, pub, return, spawn, tool, true, try, type, use, while

Literals:
- Integer: 0, 42, 1_000 (underscore separators allowed).
- Float: 3.14, 1.0e-3.
- String: "text" with escapes \n \t \" \\.
- Bool: true, false.
- None: none (for optional types).

Operators and punctuation:
- Block delimiters: ::
- Binding/assignment: :=
- Named argument: =
- Arrows: ->, =>
- Error propagation: ?
- Arithmetic: + - * / %
- Comparison: < <= > >= == !=
- Logical: and or not
- Access: .
- Delimiters: ( ) [ ] { } , : ;

Note: "=" is only used for named arguments and configuration fields, not for
binding or assignment.

-------------------------------------------------------------------------------
2. "::" blocks and validation
-------------------------------------------------------------------------------

Block start:
- A block begins when a line ends with "::" after a block header.

Block end:
- A block ends with a line that contains only "::" (ignoring whitespace and
  comments).

Validation rules:
- The parser maintains a stack of open blocks.
- Each BLOCK_START pushes; each BLOCK_END pops.
- BLOCK_END when the stack is empty is an error.
- EOF with a non-empty stack is an error.
- A BLOCK_END line may not contain additional tokens.

Example:
fn greet(name: String) -> String ::
    return "Hi " + name
::

-------------------------------------------------------------------------------
3. Grammar (EBNF)
-------------------------------------------------------------------------------

Notation:
- ? optional, * repetition, | alternatives
- BLOCK_START and BLOCK_END are produced by the lexer using section 2 rules.

module        = { item } EOF ;
item          = use_decl
              | policy_decl
              | tool_decl
              | prompt_decl
              | model_decl
              | agent_decl
              | type_decl
              | enum_decl
              | impl_decl
              | fn_decl
              | stmt ;

use_decl      = [ "pub" ] "use" module_path [ use_list ] [ "as" ident ] stmt_end ;
use_list      = "::" "{" ident { "," ident } "}" ;
module_path   = ident { "." ident } ;

fn_decl       = [ "pub" ] [ "async" ] "fn" ident "(" param_list? ")"
                [ "->" type_ref ] block ;
param_list    = param { "," param } ;
param         = ident ":" type_ref [ "=" expr ] ;

type_decl     = [ "pub" ] "type" ident type_params? block ;
enum_decl     = [ "pub" ] "enum" ident type_params? block ;
impl_decl     = "impl" ident type_params? block ;
type_params   = "<" ident { "," ident } ">" ;

tool_decl     = "tool" tool_path "(" param_list? ")"
                [ "->" type_ref ] stmt_end ;
tool_path     = ident { "." ident } ;

policy_decl   = "policy" ident block ;
policy_rule   = ("allow" | "deny") capability rule_filters? stmt_end ;
capability    = ident { "." ident } ;
rule_filters  = filter { "," filter } ;
filter        = ident "=" literal | ident "=" list_lit ;

prompt_decl   = "prompt" ident block ;
model_decl    = "model" ident ":=" expr stmt_end ;

agent_decl    = "agent" ident block_agent ;
block_agent   = BLOCK_START { agent_item } BLOCK_END ;
agent_item    = policy_use | memory_decl | fn_decl | stmt ;
policy_use    = "policy" ident stmt_end ;

memory_decl   = "memory" ident "(" string_lit ")" stmt_end
              | "memory" ident ":=" expr stmt_end ;

block         = BLOCK_START { stmt } BLOCK_END ;
stmt          = let_stmt
              | assign_stmt
              | if_stmt
              | for_stmt
              | while_stmt
              | match_stmt
              | try_stmt
              | return_stmt
              | break_stmt
              | continue_stmt
              | expr stmt_end ;

stmt_end      = ";" | NL ;

let_stmt      = "let" ident type_annot? ":=" expr stmt_end ;
type_annot    = ":" type_ref ;

assign_stmt   = lvalue ":=" expr stmt_end ;
lvalue        = ident { "." ident | "[" expr "]" } ;

if_stmt       = "if" expr block [ "else" (block | if_stmt) ] ;
for_stmt      = "for" ident "in" expr block ;
while_stmt    = "while" expr block ;
match_stmt    = "match" expr block_match ;
block_match   = BLOCK_START { match_arm } BLOCK_END ;
match_arm     = pattern "=>" (expr stmt_end | block) ;

try_stmt      = "try" block "catch" ident block ;
return_stmt   = "return" expr? stmt_end ;
break_stmt    = "break" stmt_end ;
continue_stmt = "continue" stmt_end ;

pattern       = "_" | literal | ident ;

expr          = assign ;
assign        = or_expr [ ":=" assign ] ;
or_expr       = and_expr { "or" and_expr } ;
and_expr      = eq_expr { "and" eq_expr } ;
eq_expr       = cmp_expr { ("==" | "!=") cmp_expr } ;
cmp_expr      = add_expr { ("<" | "<=" | ">" | ">=") add_expr } ;
add_expr      = mul_expr { ("+" | "-") mul_expr } ;
mul_expr      = unary { ("*" | "/" | "%") unary } ;
unary         = { "not" | "-" | "await" | "spawn" } postfix ;
postfix       = primary { call | index | field | "?" } ;
call          = "(" arg_list? ")" ;
arg_list      = arg { "," arg } ;
arg           = expr | ident "=" expr ;
index         = "[" expr "]" ;
field         = "." ident ;

primary       = literal
              | ident
              | "(" expr ")"
              | list_lit
              | lambda
              | match_expr ;

match_expr    = "match" expr block_match ;
lambda        = "(" param_list? ")" [ "->" type_ref ] "=>" expr ;

list_lit      = "[" [ expr { "," expr } ] "]" ;

literal       = int_lit | float_lit | string_lit | "true" | "false" | "none" ;

type_ref      = fn_type | named_type ;
fn_type       = "fn" "(" [ type_ref { "," type_ref } ] ")" "->" type_ref ;
named_type    = ident type_args? "?"? ;
type_args     = "<" type_ref { "," type_ref } ">" ;

-------------------------------------------------------------------------------
4. Operator precedence (highest to lowest)
-------------------------------------------------------------------------------

1) Postfix: field access, call, index, error propagation (., (), [], ?)
2) Unary: not, -, await, spawn
3) Multiplicative: * / %
4) Additive: + -
5) Comparison: < <= > >=
6) Equality: == !=
7) Logical and
8) Logical or
9) Assignment: :=

-------------------------------------------------------------------------------
5. Module and package layout
-------------------------------------------------------------------------------

File extension:
- .morph

Package layout:
- morph.toml
- morph.lock
- src/main.morph (entry)
- src/lib.morph (library entry)
- src/<path>.morph for modules

Module resolution:
- use foo.bar resolves to src/foo/bar.morph or src/foo/bar/index.morph.
- A file defines a module with the same path.

Exports and visibility (v0.3):
- Items are private unless declared with pub.
- pub applies to fn, type, enum, and use only.
- pub use re-exports a module or symbol.
- Re-export syntax: pub use foo.bar or pub use foo.bar::{add, sub}.
- use lists do not support aliases.
- use foo.bar imports a module when foo.bar resolves to a module file.
- If the full path does not resolve to a module file, the last segment is
  treated as a symbol in the parent module path.
- Importing a private symbol is an error.

Package manifest (morph.toml):
[package]
name = "morph_app"
version = "0.1.0"

[dependencies]
net = "0.3"

Local path dependencies (v0.3):
[dependencies]
utils = { path = "../utils" }
- Dependency module root name is the dependency key (utils -> that dep's src/).

-------------------------------------------------------------------------------
6. Standard library design (v1.0)
-------------------------------------------------------------------------------

Core modules:
- std.core: prelude types, Result, Option, basic traits.
- std.collections: List, Map, Set, iterators.
- std.math: numeric ops, random.
- std.string: string helpers, regex.
- std.io: stdin, stdout, streams.
- std.fs: files, paths (capability-gated).
- std.net.http: HTTP client and server (capability-gated).
- std.net.tcp: sockets (capability-gated).
- std.time: clocks, timers, sleep.
- std.crypto: hashes, HMAC, key utils.
- std.json: JSON encoding/decoding.
- std.async: tasks, channels, timers.
- std.ai: prompt, model, memory helpers.

-------------------------------------------------------------------------------
7. AI-native features
-------------------------------------------------------------------------------

Tools:
- tool declarations define typed, permissioned host functions.
- Tools are invoked like functions and are capability-gated.

Agents:
- agent blocks define a policy, memory, and methods.
- agent functions can call tools; permissions are enforced by policy.

Policies:
- policy blocks define allow/deny rules for capabilities.
- Default is deny; explicit allow is required.

Memory:
- memory declarations create persistent stores (vector, key-value, file).
- memory operations are capability-gated (memory.read, memory.write).

Prompts:
- prompt defines typed input/output and a template block.
- prompt usage is type-checked at compile time.

Models:
- model declarations configure LLM backends and inference parameters.

-------------------------------------------------------------------------------
8. "::" nesting validation algorithm
-------------------------------------------------------------------------------

Given the token stream and line info:
1) For each line, if the last non-whitespace token is "::" and the line has a
   block header token before it, emit BLOCK_START.
2) If a line contains only "::", emit BLOCK_END.
3) Track a stack count. Error on BLOCK_END when count is 0.
4) Error on EOF when count is not 0.

-------------------------------------------------------------------------------
9. Sandboxing and capability enforcement
-------------------------------------------------------------------------------

Capabilities (examples):
- fs.read, fs.write
- net.connect, net.listen
- process.exec
- env.read, env.write
- tool.*
- model.invoke
- memory.read, memory.write

Rules:
- Policies are attached to modules, agents, or blocks.
- Effective policy is the intersection of active policies.
- All IO, network, model, tool, and FFI calls require explicit allow.
- A denied operation raises PolicyError at runtime.

Filters (v0.3):
- path_prefix="...": paths are normalized ("/" vs "\\", collapse "..") and
  compared as prefix; Windows comparisons are case-insensitive.
- domain="...": matches if the request domain ends with the filter value.
- Filters may be specified as lists; any matching value satisfies the filter.

Enforcement:
- Compiler emits a manifest of required capabilities per module.
- Runtime enforces allow/deny checks at each capability boundary.
- Host can provide a global policy to further restrict execution.

-------------------------------------------------------------------------------
10. Notes and open items
-------------------------------------------------------------------------------

- Map/Set literal syntax is optional in v1.0 (use std.collections constructors).
- Pattern matching can be expanded with destructuring in later versions.
- Diagnostics include line/col, snippets, and labeled spans where possible.
- Diagnostics output contract (v0.3):
  error: <message>
  --> file:line:col
  <line_no> | <source line>
            | ^ label
