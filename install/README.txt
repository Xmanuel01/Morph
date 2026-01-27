Enkai Quickstart

1) Verify installation:
   enkai --version

2) Run the hello example:
   enkai run examples/hello/main.enk

3) Create your own program:
   echo "print(\"Hello Enkai\")" > hello.enk
   enkai run hello.enk

Notes:
- The std/ folder must remain next to the enkai executable.
- If native modules are included, keep the enkai_native library in the same folder.

