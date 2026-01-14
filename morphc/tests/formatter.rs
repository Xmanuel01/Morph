use morphc::formatter::{check_format, format_source};

#[test]
fn formats_block_indentation() {
    let source = "fn main() -> Int ::\nprint(\"hi\")\n::\n";
    let formatted = format_source(source).expect("format");
    assert!(formatted.contains("\n    print(\"hi\")\n"));
}

#[test]
fn detects_unformatted_source() {
    let source = "if true ::\nprint(\"hi\")\n::\n";
    assert!(check_format(source).is_err());
}

#[test]
fn accepts_formatted_source() {
    let source = "if true ::\n    print(\"hi\")\n::\n";
    assert!(check_format(source).is_ok());
}
