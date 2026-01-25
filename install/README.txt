Morph Quickstart

1) Verify installation:
   morph --version

2) Run the hello example:
   morph run examples/hello/main.morph

3) Create your own program:
   echo "print(\"Hello Morph\")" > hello.morph
   morph run hello.morph

Notes:
- The std/ folder must remain next to the morph executable.
- If native modules are included, keep the morph_native library in the same folder.