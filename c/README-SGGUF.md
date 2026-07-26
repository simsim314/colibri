# SGGUF bitmap sparse experts

The current converter writes the `SPB3` sparse tensor payload. Routed MoE
expert tensors are divided into logical groups of at most 256 weights. A full
group is stored as:

```text
32-byte occupancy bitmap
fixed codec auxiliary data
retained quantized codes, packed at their original bit width
```

There is no tree, per-group header, or per-group alignment padding. A direct
32-bit relative-offset table locates each variable-length group in constant
time. `SPB3` differs from `SPB2` by supporting a short final group when a row
length is not divisible by 256. This is required by GPT-OSS, whose expert rows
are 2880 values (`11 * 256 + 64`).

The outer SGGUF file remains mixed storage:

- non-routed tensors are copied byte-for-byte from the source GGUF;
- routed MoE gate/up/down tensors use SPB3 bitmap storage;
- ordinary GGUF models continue through the dense runtime path;
- legacy SPT1 tree and SPB2 bitmap SGGUF files remain readable.

## Exact sparse codecs

The converter preserves the original quantized code and scale metadata for
retained values whenever an exact codec exists:

```text
Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1
Q3_K Q4_K Q5_K Q6_K Q8_K
IQ4_XS MXFP4
F16 BF16 F32 retained-value streams
```

For GPT-OSS MXFP4, each 256-value group stores eight original E8M0 scale bytes
plus retained four-bit E2M1 codes. For IQ4_XS, each group stores the original
FP16 super-scale and packed subgroup scales plus retained nonlinear four-bit
codes. The runtime computes directly from the bitmap and packed values; it
does not reconstruct a persistent dense tensor.

## Convert

```bash
./c/sgguf-convert INPUT.gguf OUTPUT.sgguf \
  --threshold 0.01 \
  --codec auto \
  --mp auto \
  --max-output-gb 48
```

`--max-output-gb` removes the temporary output and fails if the generated file
crosses the requested limit. The converter semantically verifies sampled
sparse blocks before renaming the temporary file. `--no-verify` disables that
validation.

## Inspect GGUF or SGGUF

```bash
./c/sgguf-inspect MODEL.gguf
./c/sgguf-inspect MODEL.sgguf --tensors
```

The report includes architecture, tensor types, routed-expert counts and
bytes, unsupported tensor types, and sparse payload accounting.

## Run

Use the same model argument for GGUF and SGGUF:

```bash
./c/colibri --gguf MODEL.sgguf --device cuda ...
```

CPU kernels enumerate bitmap set bits. CUDA uploads the selected compressed
block range plus its normalized offset table and decodes retained codes inside
the sparse matvec kernel. Dense GGUF IQ4_XS and MXFP4 tensors use the ordinary
dense quantized matvec path.

## Format limit

Each sparse tensor's block stream uses 32-bit relative offsets and therefore
must remain below 4 GiB. The limit applies per sparse tensor, not to the whole
SGGUF file.
