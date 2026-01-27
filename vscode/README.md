# Enkai VS Code Extension

This folder contains the VS Code extension for the Enkai language. It provides:

- Syntax highlighting
- Basic indentation + folding for `::` blocks
- Code snippets for common Enkai constructs

## Install (local)

1. Open this repo in VS Code.
2. Run `code --extensionDevelopmentPath=./vscode` to launch an Extension Host.
3. Open any `.enk` file to see highlighting and snippets.

## Package (optional)

If you want a VSIX for distribution:

```
cd vscode
npx @vscode/vsce package
```

This creates a `.vsix` file you can install from VS Code.

