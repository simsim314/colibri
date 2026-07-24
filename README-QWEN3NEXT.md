# Qwen3-Next GGUF

Native Qwen3-Next inference with Colibrì’s GGUF tensor backend and expert scheduler.

## Build

```bash
cd ~/GitHub/colibri/c

make clean
make colibri CUDA=1 CUDA_ARCH=61 -j"$(nproc)"
```

## Model

```bash
MODEL='/home/simsim314/Downloads/Qwen3-Next-80B-A3B-Thinking-GRPO-Uncensored-Q4_K_M.gguf'
```

## Recommended GTX 1060 configuration

This is the released-compatible execution path and currently performs best on the 6 GB GTX 1060:

```bash
QWEN_ROUTER_GPU=0 \
PILOT=0 \
PILOT_REAL=0 \
PIN=auto \
AUTOPIN=1 \
CUDA_RESERVE_GB=3.5 \
./colibri \
  --gguf "$MODEL" \
  --device cuda \
  --prompt $'<|im_start|>system\nYou are a concise technical assistant.<|im_end|>\n<|im_start|>user\nExplain how an MoE expert cache improves inference speed.<|im_end|>\n<|im_start|>assistant\n' \
  --max-tokens 150 \
  --verbose
```

## GPU router test

The Qwen softmax and top-10 router can run on CUDA:

```bash
QWEN_ROUTER_GPU=1 \
PILOT=0 \
PILOT_REAL=0 \
PIN=auto \
AUTOPIN=1 \
CUDA_RESERVE_GB=3.5 \
./colibri \
  --gguf "$MODEL" \
  --device cuda \
  --prompt 'Tell a short story about France.' \
  --max-tokens 60 \
  --verbose
```

On the GTX 1060, the CPU router is currently faster.

## Pilot test

```bash
QWEN_ROUTER_GPU=0 \
PILOT=1 \
PILOT_REAL=1 \
PILOT_K=4 \
PIN=auto \
AUTOPIN=1 \
CUDA_RESERVE_GB=3.5 \
./colibri \
  --gguf "$MODEL" \
  --device cuda \
  --prompt 'Explain mixture-of-experts routing.' \
  --max-tokens 60 \
  --verbose
```

When too few LRU slots are available per layer, real GPU prefetch automatically becomes an mmap page-cache hint.

## Focused hot-expert grouping

Concentrate the existing hot-pin budget in an automatically selected number of
layers and group simultaneous resident hits only inside those focused layers:

```bash
QWEN_ROUTER_GPU=0 \
QWEN_FOCUS_LAYER_COUNT=3 \
QWEN_FOCUS_GROUP=1 \
QWEN_FOCUS_GROUP_MIN=2 \
PILOT=0 \
PILOT_REAL=0 \
PIN=auto \
AUTOPIN=1 \
CUDA_RESERVE_GB=3.5 \
./colibri \
  --gguf "$MODEL" \
  --device cuda \
  --prompt 'Tell a short story about France.' \
  --max-tokens 60 \
  --verbose
```

Non-focused layers retain the released-compatible one-expert-at-a-time GPU
streaming path. The grouped kernel receives only experts that were already
resident; misses are admitted sequentially after the group completes.

Estimate a layer count from the usage history without loading the model:

```bash
python3 tools/qwen_focus_tool.py recommend \
  "$MODEL.coli_usage" \
  --pin-slots 73 \
  --max-layers 48
```

The cost model is approximate. Confirm nearby layer counts with measured tok/s.

## Important flags

| Flag                    | Meaning                                              |
| ----------------------- | ---------------------------------------------------- |
| `QWEN_ROUTER_GPU=0`     | Download 512 router logits and select top-10 on CPU  |
| `QWEN_ROUTER_GPU=1`     | Perform Qwen softmax and top-10 selection on CUDA    |
| `PIN=auto`              | Pin experts selected from recorded usage history     |
| `AUTOPIN=1`             | Enable automatic hot-expert selection                |
| `PIN_GB=0.05`           | Optionally limit the pinned-expert tier              |
| `QWEN_FOCUS_LAYER_COUNT=K` | Concentrate hot pins in K automatically chosen layers |
| `QWEN_FOCUS_GROUP=1`     | Group already-resident selected experts in focused layers |
| `QWEN_FOCUS_GROUP_MIN=2` | Minimum resident hits required for a grouped launch |
| `PILOT=0`               | Disable speculative next-layer routing               |
| `PILOT=1`               | Enable speculative expert prediction                 |
| `PILOT_REAL=1`          | Request actual speculative GPU residency             |
| `PILOT_K=4`             | Number of speculative expert candidates              |
| `PILOT_REAL_FORCE=1`    | Force real prefetch despite a very small layer cache |
| `CUDA_RESERVE_GB=3.5`   | Protect 3.5 GiB from expert-cache allocation         |
| `CUDA_EXPERT_GB=<size>` | Explicitly limit the expert-cache budget             |
| `--max-tokens N`        | Maximum generated tokens                             |
| `--verbose`             | Print scheduler and Qwen telemetry                   |

## Telemetry

A verbose run reports:

```text
[SCHED] hits=... misses=... admissions=... evictions=...
[QWEN] router=cpu|gpu
       router_gpu=...
       router_cpu_fallback=...
       expert_cpu_fallback=...
       residency_fail=...
       compute_fail=...
       admissions=enabled|disabled
       pilot=...
       group_calls=...
       group_experts=...
       group_fail=...
```

`residency_fail=0` and `compute_fail=0` indicate normal CUDA expert execution.

## Current GTX 1060 benchmark result

With the same 35-token prompt and 60 generated tokens:

```text
CPU router, auto pin, no pilot: 0.818 tok/s
CPU router, low pin:            0.761 tok/s
CPU router, all LRU:            0.663 tok/s
GPU router, all LRU:            0.610 tok/s
GPU router, auto pin:           0.599 tok/s
GPU router plus pilot hint:     0.497 tok/s
```

Recommended configuration:

```text
QWEN_ROUTER_GPU=0
PILOT=0
PILOT_REAL=0
PIN=auto
AUTOPIN=1
CUDA_RESERVE_GB=3.5
```

## Focus hot experts into selected layers

`QWEN_FOCUS_LAYER_COUNT=K` keeps the normal per-layer LRU execution slots, but
concentrates the historical hot pinned tier into exactly `K` automatically
chosen layers. `K=0` preserves the previous global hot-expert policy.

```bash
QWEN_ROUTER_GPU=0 \
QWEN_FOCUS_LAYER_COUNT=3 \
PILOT=0 \
PILOT_REAL=0 \
PIN=auto \
AUTOPIN=1 \
CUDA_RESERVE_GB=3.5 \
./colibri \
  --gguf "$MODEL" \
  --device cuda \
  --prompt 'Tell a short story about France.' \
  --max-tokens 60 \
  --verbose
```

The runtime prints the selected layers and pin counts:

```text
[FOCUS] Qwen hot pins: requested=3 effective=3 budget=73 layers=12:25,36:24,24:24
```

### Estimate a layer count from usage history

Use the pin count printed by a normal run:

```bash
python3 tools/qwen_focus_tool.py recommend \
  "$MODEL.coli_usage" \
  --pin-slots 73
```

This evaluates `K=1..48` against recorded expert selections. It is only a
history-based estimate.

### Empirically benchmark K=1..48

```bash
python3 tools/qwen_focus_tool.py benchmark \
  "$MODEL.coli_usage" \
  --model "$MODEL" \
  --binary ./colibri \
  --max-tokens 60 \
  --rounds 1 \
  --include-baseline
```

The tool restores the identical `.coli_usage` file before every run, writes all
logs and a CSV, and reports the fastest measured `K`. Running all 48 values can
take a long time; use `--start` and `--end` for a smaller range.
