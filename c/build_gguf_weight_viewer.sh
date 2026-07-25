#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

for f in gguf_reader.c gguf_reader.h ggml_types.c ggml_types.h ggml_quants.c ggml_quants.h compat.h; do
  if [[ ! -f "$f" ]]; then
    echo "missing $f; extract this ZIP inside ~/GitHub/colibri/c" >&2
    exit 1
  fi
done

gcc -O3 -march=native -fPIC -shared -D_FILE_OFFSET_BITS=64 \
  gguf_weight_bridge.c \
  gguf_reader.c \
  ggml_types.c \
  ggml_quants.c \
  -lm \
  -o libgguf_weight_bridge.so

python3 - <<'PY'
import numpy, matplotlib
print("Python dependencies: numpy", numpy.__version__, "matplotlib", matplotlib.__version__)
PY

echo "built: $(pwd)/libgguf_weight_bridge.so"
