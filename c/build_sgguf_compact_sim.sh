#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${1:-$HOME/GitHub/colibri}"
C_DIR="$REPO/c"

for f in gguf_reader.c gguf_reader.h ggml_types.c ggml_types.h ggml_quants.c ggml_quants.h compat.h; do
  [[ -f "$C_DIR/$f" ]] || { echo "missing $C_DIR/$f" >&2; exit 1; }
done

cp "$HERE/sgguf_compact_sim_bridge.c" "$C_DIR/sgguf_compact_sim_bridge.c"
cp "$HERE/sgguf_compact_layout_sim.py" "$C_DIR/sgguf_compact_layout_sim.py"

gcc -O3 -march=native -fPIC -shared -D_FILE_OFFSET_BITS=64 \
  "$C_DIR/sgguf_compact_sim_bridge.c" \
  "$C_DIR/gguf_reader.c" \
  "$C_DIR/ggml_types.c" \
  "$C_DIR/ggml_quants.c" \
  -I"$C_DIR" -lm \
  -o "$C_DIR/libsgguf_compact_sim.so"

python3 - <<'PY'
import numpy
print("numpy", numpy.__version__)
PY

echo "installed: $C_DIR/sgguf_compact_layout_sim.py"
echo "built:     $C_DIR/libsgguf_compact_sim.so"
