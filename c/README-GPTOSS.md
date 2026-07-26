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
- dense IQ4_XS and MXFP4 CPU/CUDA matvec kernels;
- sparse IQ4_XS and MXFP4 CPU/CUDA matvec kernels.

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
GPTOSS_EXPERT_CACHE_PER_LAYER=4 \
./c/colibri \
  --gguf "$OUT" \
  --device cuda \
  --context 4096 \
  --raw-prompt \
  --prompt "$HARMONY_PROMPT" \
  --max-tokens 80 \
  --verbose
```

Without `--raw-prompt`, Colibri renders a basic system/user/assistant Harmony
conversation. `GPTOSS_EXPERT_CACHE_PER_LAYER` controls selected expert views
retained per layer; it is never set below the model's active-expert count.
