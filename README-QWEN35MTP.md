# Qwen3.5/Qwen3.6 MoE + MTP GGUF

Native `qwen35moe` text inference for Colibrì, including hybrid Gated DeltaNet/full attention, 256-way top-8 MoE routing, the appended `nextn` MTP decoder block, and lossless greedy self-speculative decoding.

Vision requires a separate `mmproj` GGUF and is not part of this text-model patch.

## Build on GTX 1060

```bash
cd ~/GitHub/colibri/c
make clean
make colibri CUDA=1 CUDA_ARCH=61 -j"$(nproc)"
make cuda-test CUDA=1 CUDA_ARCH=61
```

## Model

```bash
MODEL='/home/simsim314/Downloads/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K_P-MTP.gguf'
```

## Recommended GTX 1060 configuration

On a GTX 1060 6 GB, keep MTP disabled. The extra MTP decoder block reduces the routed-expert cache and its verification overhead is larger than the small-batch speedup available on Pascal. The measured 80-token runs were:

```text
--mtp-draft 0: 1.041 tok/s
--mtp-draft 2: 0.882 tok/s (87.9% draft acceptance)
--mtp-draft 5: 0.697 tok/s (63.2% draft acceptance)
```

Use `--mtp-draft 0` as the default. This binds the MTP tensor metadata from the mmap-backed GGUF, but does not make the MTP block resident, allocate its KV cache, or execute it. In the tested model this saved 40.35 MiB of dense VRAM for the trunk expert cache.

## Correctness baseline: MTP disabled

```bash
PROMPT=$'<|im_start|>user\nExplain mixture-of-experts inference briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n'

Q35_ROUTER_GPU=0 \
PILOT=0 PILOT_REAL=0 \
PIN=auto AUTOPIN=1 \
CUDA_RESERVE_GB=3.5 \
./colibri --gguf "$MODEL" --device cuda \
  --raw-prompt \
  --prompt "$PROMPT" \
  --max-tokens 80 --mtp-draft 0 --verbose
```

## Qwen3.5/Qwen3.6 correctness fixes

The implementation includes the model-specific details required by current Qwen3.5/Qwen3.6 GGUF files:

- Native CUDA decoding for mixed-precision `Q8_0` dense tensors, including protected `attn_qkv` matrices in `Q6_K_P` models.
- Tiled value-head mapping used by current llama.cpp conversion when the model has more value heads than key heads. For the 16-key/32-value-head model, the key head is `value_head % 16`, not `value_head / 2`.
- The shared Qwen3-Next DeltaNet recurrence remains unchanged; only the Qwen3.5/Qwen3.6 GGUF head layout is handled differently.
- CUDA residency failures report tensor type and memory context, and can fall back to mmap-backed CPU execution rather than aborting.

These details are correctness-critical: using grouped value-head indexing with a tiled GGUF produces fluent-looking startup telemetry but junk tokens on both CPU and CUDA.

## MTP self-speculation

Start with one or two drafts:

```bash
Q35_ROUTER_GPU=0 \
PILOT=0 PILOT_REAL=0 \
PIN=auto AUTOPIN=1 \
CUDA_RESERVE_GB=3.5 \
./colibri --gguf "$MODEL" --device cuda \
  --prompt 'Explain mixture-of-experts inference briefly.' \
  --max-tokens 80 --mtp-draft 2 --verbose
```

`--mtp-draft N` accepts `0..8`:

- `0`: ordinary autoregressive decoding; the appended MTP block is bound but not resident or executed.
- `1..8`: recursively draft up to `N` tokens, verify `N+1` target inputs as one bounded block, accept exact greedy matches, and emit the target bonus token when all drafts match.
- `--no-mtp`: alias for `--mtp-draft 0`.
- `QWEN35_MTP_DRAFT=N`: environment alternative.

The implementation is lossless for greedy decoding: every draft is verified by the full 40-layer target trunk. A rejected block restores the recurrent DeltaNet state and replays only the accepted prefix plus the mismatch input.

Prompt prefill and verified-token catch-up use an MTP K/V-only synchronization path. They skip the MTP query/attention output, MoE block, 248,320-token vocabulary projection, and logits download. Full MTP execution is reserved for actual draft proposals. Telemetry separates `draft_steps` from `kv_updates`.

## VRAM priority and automatic fallback

CUDA residency is best-effort, not a load requirement:

1. Mandatory KV, convolution, and recurrent state is allocated first.
2. The output projection and executable trunk tensors are then made resident.
3. `token_embd.weight` is not copied wholesale when a separate output head exists; Colibrì decodes one embedding row from mmap and uploads only 8 KiB per token.
4. The MTP block and MTP scratch are allocated only when `--mtp-draft` is greater than zero.
5. Routed-expert cache slots use only the remaining VRAM after `CUDA_RESERVE_GB`.

If mandatory CUDA state or dense residency cannot fit, Colibrì releases partial CUDA allocations and continues with mmap-backed CPU inference instead of aborting. Verbose output reports:

```text
[CUDA] Q35 VRAM fallback: ...; releasing CUDA state/weights and continuing from mmap on CPU RAM
[GGUF] ... backend=cpu (automatic CUDA VRAM fallback)
```

This fallback is correctness-first. Close other CUDA processes to retain the faster GPU path.

## Useful flags

| Flag | Meaning |
|---|---|
| `Q35_ROUTER_GPU=0` | CPU top-8 selection; recommended default on Pascal |
| `Q35_ROUTER_GPU=1` | CUDA top-8 selection |
| `PIN=auto` / `AUTOPIN=1` | Use recorded expert history for hot pins |
| `CUDA_RESERVE_GB=3.5` | Keep runtime/state/scratch VRAM outside the expert-cache budget |
| `CUDA_EXPERT_GB=X` | Explicit expert residency budget |
| `PILOT=0` / `PILOT_REAL=0` | Disable speculative expert prefetch |
| `Q35_FOCUS_LAYER_COUNT=K` | Optional focused hot-pin experiment |
| `Q35_FOCUS_GROUP=0|1` | Disable/enable focused grouped resident experts |
| `--raw-prompt` | Skip Qwen chat-template formatting |
| `--verbose` | Print model, scheduler, CUDA, and MTP telemetry |

## MTP telemetry

```text
[MTP] enabled=1 draft_max=2 mode=block-verify
      steps=... proposed=... accepted=... (...%)
      verify_batches=... verify_tokens=... replays=...
```

Compare `--mtp-draft 0`, `1`, `2`, and `4` using the same prompt, expert-cache budget, and output length. Acceptance alone does not guarantee a speedup. On the tested GTX 1060, `--mtp-draft 0` is the best current setting.
