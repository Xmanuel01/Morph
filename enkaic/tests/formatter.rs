use enkaic::formatter::{check_format, format_source};

#[test]
fn formats_block_indentation() {
    let source = "fn main() -> Int ::\nprint(\"hi\")\n::\n";
    let formatted = format_source(source).expect("format");
    assert!(formatted.contains("\n    print(\"hi\")\n"));
    assert!(formatted.contains("\n::fn\n"));
}

#[test]
fn detects_unformatted_source() {
    let source = "if true ::\nprint(\"hi\")\n::\n";
    assert!(check_format(source).is_err());
}

#[test]
fn accepts_formatted_source() {
    let source = "if true ::\n    print(\"hi\")\n::if\n";
    assert!(check_format(source).is_ok());
}

#[test]
fn upgrades_nested_plain_closers_to_tagged_closers() {
    let source = "fn main() ::\nwhile ready ::\nif ready ::\nprint(\"done\")\n::\n::\n::\n";
    let formatted = format_source(source).expect("format");
    assert!(formatted.contains("\n            print(\"done\")\n"));
    assert!(formatted.contains("\n        ::if\n"));
    assert!(formatted.contains("\n    ::while\n"));
    assert!(formatted.contains("\n::fn\n"));
}

#[test]
fn upgrades_type_closer_to_struct_alias() {
    let source = "type Pair ::\nleft: Int\nright: Int\n::\n";
    let formatted = format_source(source).expect("format");
    assert!(formatted.contains("\n::struct\n"));
}

#[test]
fn rejects_mismatched_existing_tagged_closer() {
    let source = "while ready ::\n    print(\"done\")\n::if\n";
    let err = format_source(source).unwrap_err();
    assert!(err
        .message
        .contains("SyntaxError: expected ::while to close while block, found ::if"));
}
