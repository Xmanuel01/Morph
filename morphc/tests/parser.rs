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
