# Enkai VS Code Extension

This folder contains the first-party VS Code extension for the Enkai language.
It provides:

- Syntax highlighting for `.enk`, `.enkai`, and `.en` files
- Highlighting for `::` blocks and tagged closers such as `::fn`, `::if`, `::while`, and `::policy`
- Highlighting for imports, policies, native imports, functions, types, tensor/AI-native types, literals, and operators
- Basic indentation and folding for anonymous and tagged `::` blocks
- Snippets for common Enkai constructs using the official tagged-closer style

## Install Locally

From the repository root:

```powershell
code --extensionDevelopmentPath=./vscode
```

Then open any `.enk` or `.enkai` file in the Extension Host window.

## Package A VSIX

```powershell
cd vscode
npx @vscode/vsce package
```

Install the generated `.vsix` in VS Code with:

```powershell
code --install-extension .\enkai-language-0.2.0.vsix
```

## Scope

This extension is syntax/snippet-only. It does not yet include an LSP server,
diagnostics, formatter integration, hover docs, go-to-definition, or debugger
support. Those should be added as separate extension tranches.
