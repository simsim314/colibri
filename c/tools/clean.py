#!/usr/bin/env python3
"""Remove build artifacts. Used by `make clean` so it works from any shell.

Works from cmd.exe, PowerShell, Git Bash, or MSYS2 — no `rm` or POSIX
`for` loop required. Silently ignores files that don't exist.
"""
import glob
import os
import shutil

# Files (relative to c/) to remove if present.
FILES = [
    "olmoe", "olmoe.exe",
    "glm", "glm.exe",
    "iobench", "iobench.exe",
    "gguf-inspect", "gguf-inspect.exe",
    "colibri", "colibri.exe",
    "run-f32-gguf", "run-f32-gguf.exe",
    "backend_cuda.o", "backend_loader.o",
    "backend_cuda_test", "backend_cuda_test.exe",
    "backend_cuda_bench", "backend_cuda_bench.exe",
    "backend_metal.o", "backend_metal_test",
    "tests/test_gguf_reader", "tests/test_gguf_reader.exe",
    "tests/test_ggml_types", "tests/test_ggml_types.exe",
    "tests/test_ggml_quants", "tests/test_ggml_quants.exe",
    "tests/test_tensor", "tests/test_tensor.exe",
    "tests/test_f32_kernels", "tests/test_f32_kernels.exe",
    "tests/test_gguf_tokenizer", "tests/test_gguf_tokenizer.exe",
    "tests/test_gguf_granite", "tests/test_gguf_granite.exe",
    "coli_cuda.dll", "coli_cuda.lib", "coli_cuda.exp",
]
# Test binaries match this pattern. Only remove executables (.exe on Windows,
# no extension on Unix) — never .c or .py source files.
TEST_GLOBS = ["tests/test_*.exe"]
# Directories to remove.
DIRS = ["tests/__pycache__"]

removed = 0
for f in FILES:
    if os.path.exists(f):
        os.remove(f)
        removed += 1
for pattern in TEST_GLOBS:
    for f in glob.glob(pattern):
        os.remove(f)
        removed += 1
for d in DIRS:
    if os.path.isdir(d):
        shutil.rmtree(d)
        removed += 1
print(f"clean: removed {removed} files/dirs")
