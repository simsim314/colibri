# GPT-OSS GGUF and sparse SGGUF runtime

Colibri supports the `gpt-oss` GGUF architecture through `gguf_gptoss.c` and
accepts either the original mixed-quant GGUF or an SGGUF produced by
`sgguf-convert`.

Implemented model details:

- separate Q/K/V projections and biases;
- 64 query heads, 8 KV heads, and 64-value head dimensions as read from GGUF;
- YaRN rotary scaling from model metadata;
- learned attention-sink logits;
- alternating sliding-window and full causal attention;
- biased 128-way top-4 routing with selected-logit softmax;
- biased gate/up/down expert projections;
- OpenAI clamped SwiGLU activation;
- ordinary GGUF and SPB3 sparse expert tensors through the same `ColiTensor`
  interface;
- dense IQ4_NL, IQ4_XS, and MXFP4 CPU/CUDA matvec kernels;
- sparse IQ4_NL, IQ4_XS, and MXFP4 CPU/CUDA matvec kernels.

The currently inspected `gpt-oss-120b-Uncensored-xCloud.i1-IQ4_XS.gguf`
uses IQ4_NL for all 108 routed expert matrices. Its remaining tensors are a
mixture of IQ4_NL, IQ4_XS, Q5_1, Q8_0, and F32. The IQ4_NL path is therefore
the primary GPT-OSS expert path; MXFP4 support remains available for other
GGUF variants.

The CUDA mode is a correctness-first hybrid path for small GPUs. Dense layer
weights are made resident when possible and otherwise streamed. Selected
expert views use a per-layer LRU cache, while attention state and activation
logic remain host-side. The output projection currently runs on CPU to avoid
occupying most of a 6 GB device.

## Build

```bash
cd c
make clean
make colibri sgguf-convert sgguf-inspect -j"$(nproc)"
make colibri CUDA=1 CUDA_ARCH=61 -j"$(nproc)"
make cuda-sgguf-test CUDA=1 CUDA_ARCH=61
```

## Inspect source tensor types

```bash
./c/sgguf-inspect "$MODEL" --tensors
```

The inspector exits nonzero when the file contains a dtype unknown to this
runtime.

## Sparsify routed experts

```bash
./c/sgguf-convert "$MODEL" "$OUT" \
  --threshold 0.01 \
  --codec auto \
  --mp auto \
  --max-output-gb 48
```

Only tensors named `blk.N.ffn_gate_exps.weight`,
`blk.N.ffn_up_exps.weight`, and `blk.N.ffn_down_exps.weight` are sparsified.
All attention, router, bias, norm, embedding, and output tensors are copied
byte-for-byte.

## Run

For reference comparisons, supply an already rendered Harmony prompt:

```bash
GPTOSS_MMAP_READERS=12 \
GPTOSS_EXPERT_CACHE_RESERVE_MIB=384 \
./c/colibri \
  --gguf "$OUT" \
  --device cuda \
  --context 4096 \
  --raw-prompt \
  --prompt "$HARMONY_PROMPT" \
  --max-tokens 80 \
  --repeat-penalty 1.10 \
  --verbose
```

Without `--raw-prompt`, Colibri renders a basic system/user/assistant Harmony
conversation.

The default `--expert-cache-mode stats` plans the CUDA expert cache globally
from the cumulative usage file. Each `(layer, expert)` bundle is ranked greedily
by historical selections per encoded byte, and the hottest bundles are uploaded
until the available VRAM budget is filled. Cache counts therefore vary by layer.
`GPTOSS_EXPERT_CACHE_RESERVE_MIB` keeps runtime headroom free (default `384`).
`GPTOSS_EXPERT_CACHE_MIB` optionally caps the automatic cache budget.

Use `--expert-cache-mode fixed --expert-cache-per-layer N` to select the older
per-layer runtime LRU cache. For backward compatibility,
`GPTOSS_EXPERT_CACHE_PER_LAYER=N` also selects fixed mode when no CLI cache mode
is supplied. CLI flags take priority over environment variables.

In stats mode, a miss replaces the coldest evictable expert in that layer only
after its cumulative count becomes higher. In fixed mode, misses use ordinary
per-layer LRU replacement.

`GPTOSS_MMAP_READERS` enables a persistent worker pool for selected dense
experts that are still disk-backed. After routing, gate/up/down slices are
copied from mmap into temporary RAM buffers in parallel before CPU execution
or CUDA upload. Existing preload-RAM tensors and CUDA-resident experts are
skipped. The worker count is capped at `3 * top_k` (12 for GPT-OSS top-4).
This is parallel demand loading only; it does not prefetch the next layer.

`--repeat-penalty N` applies a sign-aware penalty once per unique generated
text token before greedy argmax. The default is `1.0` (disabled); values around
`1.05` to `1.15` are useful for reducing deterministic repetition without
penalizing Harmony control tokens.
