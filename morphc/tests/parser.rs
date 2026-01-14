use morphc::lexer::tokenize;
use morphc::parser::parse_module;

#[test]
fn parses_basic_block() {
    let source = "if true ::\n    let x := 1\n::\n";
    let module = parse_module(source).expect("module should parse");
    assert!(!module.items.is_empty());
}

#[test]
fn rejects_unclosed_block() {
    let source = "if true ::\n    let x := 1\n";
    assert!(tokenize(source).is_err());
}

#[test]
fn rejects_stray_block_end() {
    let source = "::\n";
    assert!(tokenize(source).is_err());
}

#[test]
fn rejects_inline_block_marker() {
    let source = "let x := 1 ::\n";
    assert!(tokenize(source).is_err());
}

#[test]
fn rejects_equal_assignment() {
    let source = "let x = 1\n";
    assert!(parse_module(source).is_err());
}

#[test]
fn parses_nested_blocks_with_comments() {
    let source = "\
if true :: // outer
    // inside comment
    while false :: // nested
        let x := 1
    :: // end nested
:: // end outer
";
    let module = parse_module(source).expect("module should parse");
    assert!(!module.items.is_empty());
}

#[test]
fn parses_else_branch() {
    let source = "\
if true ::
    let x := 1
::
else ::
    let x := 2
::
";
    let module = parse_module(source).expect("module should parse");
    assert!(!module.items.is_empty());
}

#[test]
fn parses_match_statement_with_block_arms() {
    let source = "\
match 2 ::
    1 => ::
        let x := 1
    ::
    _ => ::
        let y := 2
    ::
::
";
    let module = parse_module(source).expect("module should parse");
    assert!(!module.items.is_empty());
}

#[test]
fn rejects_match_expression_block_arm() {
    let source = "\
let x := match 1 ::
    1 => ::
        2
    ::
    _ => 0
::
";
    assert!(parse_module(source).is_err());
}

#[test]
fn allows_block_end_with_trailing_whitespace_and_comment() {
    let source = "\
if true ::
    let x := 1
::   // end block
";
    let module = parse_module(source).expect("module should parse");
    assert!(!module.items.is_empty());
}

#[test]
fn rejects_block_end_with_extra_tokens_after_colons() {
    let source = "\
if true ::
    let x := 1
:: / / not_a_comment
";
    assert!(tokenize(source).is_err());
}

#[test]
fn rejects_block_start_with_extra_tokens_after_colons() {
    let source = "\
if true :: / / not_a_comment
    let x := 1
::
";
    assert!(tokenize(source).is_err());
}

#[test]
fn parse_error_includes_snippet() {
    let source = "let x = 1\n";
    let err = parse_module(source).expect_err("should fail");
    let message = err.to_string();
    assert!(message.contains("error: Use ':='"));
    assert!(message.contains("--> 1:7"));
    assert!(message.contains("| let x = 1"));
    assert!(message.contains("^ here"));
}

#[test]
fn parse_error_handles_crlf() {
    let source = "let x = 1\r\n";
    let err = parse_module(source).expect_err("should fail");
    let message = err.to_string();
    assert!(message.contains("--> 1:7"));
    assert!(message.contains("| let x = 1"));
    assert!(!message.contains('\r'));
}

#[test]
fn parses_use_list() {
    let source = "\
use math::{add, sub}
fn main() -> Int ::
    return 0
::
";
    let module = parse_module(source).expect("module should parse");
    assert!(!module.items.is_empty());
}

#[test]
fn rejects_pub_on_non_exportable_item() {
    let sources = [
        "pub tool web.search(query: String) -> String\n",
        "pub policy default ::\n    allow io.print\n::\n",
        "pub prompt Ask ::\n    input ::\n        topic: String\n    ::\n::\n",
        "pub model m := 1\n",
        "pub agent Bot ::\n::\n",
    ];
    for source in sources {
        let err = parse_module(source).expect_err("should fail");
        assert!(err
            .to_string()
            .contains("Only fn, type, enum, and use can be public"));
    }
}

#[test]
fn rejects_use_list_alias() {
    let source = "pub use foo::{bar} as baz\n";
    let err = parse_module(source).expect_err("should fail");
    assert!(err
        .to_string()
        .contains("Alias is not supported for use lists"));
}

#[test]
fn rejects_empty_use_list() {
    let source = "use foo::{}\n";
    let err = parse_module(source).expect_err("should fail");
    assert!(err.to_string().contains("Expected symbol in use list"));
}

#[test]
fn sets_pub_flags_for_decls() {
    let source = "\
pub fn pub_fn() -> Int ::
    return 1
::
fn priv_fn() -> Int ::
    return 0
::
pub type PubType ::
    value: Int
::
type PrivType ::
    value: Int
::
pub enum PubEnum ::
    A
::
enum PrivEnum ::
    B
::
pub use foo.bar
use foo.baz
";
    let module = parse_module(source).expect("module should parse");
    let mut saw_pub_fn = false;
    let mut saw_priv_fn = false;
    let mut saw_pub_type = false;
    let mut saw_priv_type = false;
    let mut saw_pub_enum = false;
    let mut saw_priv_enum = false;
    let mut saw_pub_use = false;
    let mut saw_priv_use = false;
    for item in module.items {
        match item {
            morphc::ast::Item::Fn(decl) if decl.name == "pub_fn" => {
                assert!(decl.is_pub);
                saw_pub_fn = true;
            }
            morphc::ast::Item::Fn(decl) if decl.name == "priv_fn" => {
                assert!(!decl.is_pub);
                saw_priv_fn = true;
            }
            morphc::ast::Item::Type(decl) if decl.name == "PubType" => {
                assert!(decl.is_pub);
                saw_pub_type = true;
            }
            morphc::ast::Item::Type(decl) if decl.name == "PrivType" => {
                assert!(!decl.is_pub);
                saw_priv_type = true;
            }
            morphc::ast::Item::Enum(decl) if decl.name == "PubEnum" => {
                assert!(decl.is_pub);
                saw_pub_enum = true;
            }
            morphc::ast::Item::Enum(decl) if decl.name == "PrivEnum" => {
                assert!(!decl.is_pub);
                saw_priv_enum = true;
            }
            morphc::ast::Item::Use(decl)
                if decl.path == vec!["foo".to_string(), "bar".to_string()] =>
            {
                assert!(decl.is_pub);
                saw_pub_use = true;
            }
            morphc::ast::Item::Use(decl)
                if decl.path == vec!["foo".to_string(), "baz".to_string()] =>
            {
                assert!(!decl.is_pub);
                saw_priv_use = true;
            }
            _ => {}
        }
    }
    assert!(saw_pub_fn);
    assert!(saw_priv_fn);
    assert!(saw_pub_type);
    assert!(saw_priv_type);
    assert!(saw_pub_enum);
    assert!(saw_priv_enum);
    assert!(saw_pub_use);
    assert!(saw_priv_use);
}
