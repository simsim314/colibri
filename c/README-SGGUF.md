# SGGUF bitmap sparse experts

The current converter writes the `SPB2` sparse tensor payload. Routed MoE
expert tensors are divided into logical groups of 256 weights. Each group is
stored as:

```text
32-byte occupancy bitmap
fixed codec auxiliary data
retained quantized codes, packed at their real bit width
```

There is no tree, no per-block header, and no per-block alignment padding.
A direct 32-bit offset table locates every variable-length group. The offset
index is memory mapped and permits constant-time access without scanning
previous groups.

For Q6_K, every retained code occupies exactly six bits. For Q8_0, every
retained code occupies eight bits. Quantization scales and other required
codec metadata are retained once per logical 256-weight group.

The outer SGGUF file remains mixed storage:

- non-routed tensors are copied byte-for-byte from GGUF;
- routed MoE gate/up/down tensors use SPB2 bitmap storage;
- ordinary `.gguf` models continue through the original dense path;
- legacy SPT1 tree SGGUF files remain readable.

## Convert

```bash
./c/sgguf-convert INPUT.gguf OUTPUT.sgguf \
  --threshold 0.01 \
  --codec auto \
  --mp auto
```

The converter validates sampled decoded values before renaming the temporary
output. `--no-verify` disables that validation.

## Inspect

```bash
./c/sgguf-inspect OUTPUT.sgguf
```

The report separates bitmap, retained-value, quantization-auxiliary, and
32-bit direct-index storage.

## Run

Use the same model option as GGUF:

```bash
./c/colibri --gguf OUTPUT.sgguf --device cuda ...
```

CPU kernels enumerate only bitmap set bits. CUDA uploads the compressed group
records plus a normalized offset table and computes directly from retained
codes; it does not reconstruct a dense expert tensor.

## Format limits

The SPB2 block stream for one sparse tensor is addressed by 32-bit relative
offsets and therefore must remain below 4 GiB. This is well above the routed
expert tensor size in the target Qwen3.6 model. A future format revision can
add 64-bit offsets for unusually large individual tensors.
