# Colibri capabilities, command surfaces, flags, and experiments

This is the master operating map for Colibri's current runtime surfaces.

**Verification scope.** This revision was checked against the latest source archive available in
this chat, `colibri-source-20260726-225803`, plus the subsequently applied GPT-OSS cache,
parallel-reader, Harmony-output, and Qwen3.5/Qwen3.6 repeat-penalty/input-echo patches. The
previous document named `colibri-source-20260727-014449`, but that archive was not supplied here,
so a newer local checkout may contain additional changes. The source and `--help` output remain
authoritative.

The most important fact is that the repository contains **two different model-execution interfaces**:
the GLM snapshot runtime and the direct GGUF runtime. Utility binaries are a third command surface.
Do not mix their options.

| Surface | Model input | Main purpose | Invocation |
|---|---|---|---|
| `c/coli` Python launcher | Colibri GLM snapshot **directory** | chat, API server, web UI, resource planning, benchmark, conversion | `./c/coli <subcommand> ...` |
| `c/colibri` native GGUF runtime | `.gguf`, `.sgguf`, or split-primary file | direct one-shot inference for supported GGUF architectures | `./c/colibri --gguf MODEL --prompt ...` |
| Native GLM engine mode | Colibri GLM snapshot **directory** | low-level research and environment-variable experiments | `SNAP=DIR ./c/colibri [cap [expert_bits [dense_bits]]]` |
| Utility binaries | GGUF/SGGUF files | inspect, sparsify, plan, and split storage | `./c/sgguf-*`, `./c/gguf-inspect` |

`coli` flags do **not** automatically apply to the direct GGUF interface. For example,
`coli --ram`, `--auto-tier`, `chat`, and `serve` belong to the GLM snapshot runtime;
the current GGUF interface is a separate one-shot command with model-specific flags.

---

## 1. Build map

Run from the repository root unless the command says otherwise.

### CPU build

```bash
cd ~/GitHub/colibri/c
make clean
make colibri -j"$(nproc)"
```

### GTX 1060 / Pascal CUDA build

```bash
cd ~/GitHub/colibri/c
make clean
make colibri CUDA=1 CUDA_ARCH=61 -j"$(nproc)"
make cuda-test CUDA=1 CUDA_ARCH=61
```

### Build all GGUF/SGGUF utilities

```bash
cd ~/GitHub/colibri/c
make gguf-inspect sgguf-convert sgguf-inspect sgguf-plan sgguf-split -j"$(nproc)"
```

### Other backends and build variables

| Build setting | Meaning |
|---|---|
| `CUDA=1` | Linux CUDA backend, linked directly with CUDA runtime. |
| `CUDA_ARCH=61` | Build specifically for GTX 1060 / sm_61. |
| `CUDA_ARCH=native` | Build for the GPU in the current machine. |
| `CUDA_ARCH=portable` | Multi-architecture release build defined by the Makefile. |
| `HIP=1 HIP_ARCH=gfx...` | Build the same GPU backend through ROCm/HIP on Linux. |
| `METAL=1` | Apple Metal backend on macOS. |
| `ARCH=native` | CPU binary optimized for the current CPU. |
| `ARCH=x86-64-v3` | Portable AVX2-class x86-64 binary. |
| `make portable` | Build using the Makefile's portable architecture baseline. |
| `make test` | C and Python tests. |
| `make check` | Clean, portable build, then all dependency-free tests. |
| `make efficiency` | Tiny-model efficiency regression suite. |
| `make efficiency-report` | Optional full-model diagnostic; not a CI gate. |
| `make install PREFIX=/usr/local` | Install `coli` and runtime support files. |

Windows uses `CUDA_DLL=1` plus `make cuda-dll`; macOS CUDA is not supported.

---

## 2. Supported inference capabilities

### Direct GGUF architectures

The native dispatcher reads `general.architecture` and currently selects:

| GGUF architecture | Runtime file | Major capabilities |
|---|---|---|
| `granitemoe` | `c/gguf_granite.c` | Granite MoE, CPU/CUDA, scheduler, hot pins, pilot/coupling experiments. |
| `qwen3next` | `c/gguf_qwen3next.c` | Qwen3-Next hybrid DeltaNet/full attention, routed MoE, CPU/CUDA. |
| `qwen35moe` | `c/gguf_qwen35moe.c` | Qwen3.5/Qwen3.6 hybrid text models, routed MoE, optional MTP self-speculation. |
| `gpt-oss` | `c/gguf_gptoss.c` | GPT-OSS dense attention + top-4 MoE, GGUF/SGGUF, cache planning, detailed tracing. |

Vision/mmproj execution is not implemented in the Qwen3.5/Qwen3.6 text path.

### Direct-runtime feature matrix

| Capability | Granite | Qwen3-Next | Qwen3.5/3.6 | GPT-OSS |
|---|:---:|:---:|:---:|:---:|
| CPU/CUDA inference | yes | yes | yes | yes |
| Colibri pin + per-layer LRU scheduler | yes | yes | yes | separate GPT-OSS cache implementation |
| MTP self-speculation | — | — | yes | — |
| `--repeat-penalty` | — | — | yes | yes |
| Formatted input echoed token-by-token | — | — | yes | yes |
| Usage-file CLI override | — | — | — | yes |
| Parallel mmap-to-RAM demand reads | — | — | — | `GPTOSS_MMAP_READERS` |
| Stats-greedy or fixed GPU cache modes | — | — | — | yes |
| Structured execution trace | — | — | — | yes |

### Storage formats

| Format | Meaning |
|---|---|
| GGUF | Ordinary dense or quantized GGUF. |
| SGGUF | Mixed file where routed expert tensors may use exact bitmap sparse `SPB3` storage. |
| GPT-OSS three-file split | Fast primary + preload shard + slow tail shard, presented as one logical model. The current hot-split planner requires GPT-OSS metadata. |
| GLM Colibri snapshot | Directory of converted shards used by the original `coli`/GLM runtime, not GGUF. |

### Quantized tensor support documented in this snapshot

The GGUF/SGGUF tensor layer includes ordinary and sparse paths for common GGML
quantizations, including Q4/Q5/Q6/Q8 families, IQ4_NL, IQ4_XS, and MXFP4.
`sgguf-convert` preserves original codes/scales for supported exact sparse codecs.

---

## 3. Direct GGUF command

### Generic form

```bash
./c/colibri --gguf "$MODEL" \
  --device cuda \
  --prompt 'Explain mixture-of-experts inference.' \
  --max-tokens 100 \
  --verbose
```

The leading `--gguf` is optional when the first argument ends in `.gguf`. Keep it for
clarity and for `.sgguf`/split-storage paths.

### Common direct-GGUF flags

| Flag | Value | Default | Meaning |
|---|---:|---:|---|
| `--gguf MODEL` | path | required | Select direct GGUF runtime. |
| `--prompt TEXT` | string | required | User text or fully rendered raw prompt. |
| `--max-tokens N` | integer | `24` | Maximum generated tokens. |
| `--device cpu` | — | `cpu` | CPU execution. |
| `--device cuda` | — | — | CUDA device 0. |
| `--device cuda:N` | integer | — | Specific CUDA device. |
| `--raw-prompt` | switch | off | Skip architecture-specific prompt formatting. |
| `--verbose`, `-v` | switch | off | Print model, residency, cache, and throughput telemetry. |

The direct GGUF paths currently use greedy decoding. They are not the same sampler/API
surface as `coli run/chat/serve`.

### Model-specific flag matrix

| Flag | Granite | Qwen3-Next | Qwen3.5/3.6 | GPT-OSS | Meaning |
|---|:---:|:---:|:---:|:---:|---|
| `--context N` | — | — | — | yes | KV context allocation. |
| `--usage-file PATH` | — | — | — | yes | Override cumulative expert-use sidecar. |
| `--repeat-penalty X` | — | — | yes | yes | Sign-aware penalty over previously generated non-control tokens; `1.0` disables. Decoding remains greedy after the logits are adjusted. |
| `--expert-cache-mode stats|fixed` | — | — | — | yes | Global statistics-ranked cache or fixed per-layer LRU. |
| `--expert-cache-per-layer N` | — | — | — | yes | Fixed cache slots per layer; implies fixed mode unless mode supplied. |
| `--mtp-draft 0..8` | — | — | yes | accepted/ignored | Qwen MTP draft count; GPT-OSS parser accepts it only for dispatcher compatibility. |
| `--no-mtp` | — | — | yes | accepted/ignored | Alias for Qwen `--mtp-draft 0`. |
| `--debug` | — | — | — | yes | Full structured execution trace. |
| `--debug-light` | dispatcher | dispatcher | dispatcher | yes | Lighter GPT-OSS debug/progress mode. |
| `--debug-dir DIR` | — | — | — | yes | Trace output directory; implies `--debug`. |

Do not pass a model-specific flag to another architecture: its parser may reject it even
though the top-level dispatcher knows the spelling.

### Output-stream behavior

- **Qwen3.5/Qwen3.6:** the runtime echoes the exact formatted input token-by-token after each
  prefill step, then continues directly with generated output. Control/template tokens are shown.
- **GPT-OSS:** the formatted Harmony input is echoed token-by-token. Generated Harmony separators
  are also printed. `<|end|>` ends one Harmony message but does **not** terminate generation;
  generation stops on EOS, `<|return|>`, `<|call|>`, or `--max-tokens`.
- **Granite and Qwen3-Next:** generated non-control text is printed; there is no equivalent input
  echo in the verified source.

---

## 4. Recommended direct-GGUF recipes

### Qwen3-Next on GTX 1060

```bash
MODEL='/path/to/Qwen3-Next.gguf'

QWEN_ROUTER_GPU=0 \
PILOT=0 \
PILOT_REAL=0 \
PIN=auto \
AUTOPIN=1 \
CUDA_RESERVE_GB=3.5 \
./c/colibri --gguf "$MODEL" --device cuda \
  --prompt 'Explain expert caching briefly.' \
  --max-tokens 100 --verbose
```

### Qwen3.5/Qwen3.6 on GTX 1060, correctness-first

```bash
MODEL='/path/to/Qwen3.6-MTP.gguf'

Q35_ROUTER_GPU=0 \
PILOT=0 PILOT_REAL=0 \
PIN=auto AUTOPIN=1 \
CUDA_RESERVE_GB=3.5 \
./c/colibri --gguf "$MODEL" --device cuda \
  --prompt 'Explain expert caching briefly.' \
  --max-tokens 100 \
  --repeat-penalty 1.10 \
  --mtp-draft 0 \
  --verbose
```

On the measured GTX 1060 path in this snapshot, MTP draft verification was slower than
ordinary autoregressive decoding despite high acceptance. Test `0`, `1`, `2`, and `4`
on the same prompt before enabling it permanently.

### GPT-OSS on CUDA

```bash
MODEL='/path/to/gpt-oss.gguf'

GPTOSS_MMAP_READERS=12 \
GPTOSS_EXPERT_CACHE_RESERVE_MIB=384 \
./c/colibri --gguf "$MODEL" --device cuda \
  --context 4096 \
  --prompt 'Write three short sentences about Paris.' \
  --max-tokens 80 \
  --repeat-penalty 1.10 \
  --expert-cache-mode stats \
  --verbose
```

For a new or weak usage file, stats mode may preload no dynamic experts. Use fixed mode while
collecting representative routing history, then compare it with stats mode:

```bash
GPTOSS_MMAP_READERS=12 \
./c/colibri --gguf "$MODEL" --device cuda \
  --usage-file "$MODEL.coli_usage" \
  --expert-cache-mode fixed \
  --expert-cache-per-layer 4 \
  --prompt 'Collect representative routing statistics.' \
  --max-tokens 100 \
  --repeat-penalty 1.10 \
  --verbose
```

### GPT-OSS full trace

```bash
./c/colibri --gguf "$MODEL" --device cuda \
  --prompt 'Test prompt' --max-tokens 4 \
  --debug --debug-dir /tmp/gptoss-debug
```

Outputs include `run.log`, per-token logs, and `summary.log`.

---

## 5. Shared GGUF scheduler and CUDA environment variables

Set variables on the command line as `NAME=value command`, or export them first.
CLI flags override the equivalent environment setting where an equivalent exists.

### Common scheduler variables for Granite/Qwen GGUF paths

| Variable | Default | Status | Meaning |
|---|---:|---|---|
| `CUDA_RESERVE_GB` | `0.5` in Granite/Qwen GGUF | normal | VRAM kept outside expert slot allocation. |
| `CUDA_EXPERT_GB` | remaining free VRAM | normal | Explicit expert-cache budget in decimal GB. |
| `PIN=auto` | unset | normal | Seed permanent hot experts from `MODEL.coli_usage`, then `stats.txt` fallback. |
| `PIN=PATH` | unset | normal | Seed pins from a specified expert-use file. |
| `PIN_GB=X` | half available slots when unspecified | normal | Limit hot pinned tier. `PIN_GB=all` fills all pin-capable slots. |
| `AUTOPIN=0|1` | `1` | normal | Automatically pin after enough history when `PIN` is not explicitly set. |
| `REPIN=N` | `0` | experimental | Re-evaluate/promote hot experts every N generated tokens. |
| `PILOT=0|1` | `0` | experimental | Predict next-layer experts; hint mode unless real loading is enabled. |
| `PILOT_REAL=0|1` | `0` | experimental | Perform actual speculative expert residency; implies `PILOT=1`. |
| `PILOT_K=N` | `8`, or `6` with real pilot | experimental | Number of predicted experts. |
| `PILOT_EVICT_GUARD=0|1` | `1` | normal safeguard | Prevent speculative loads from evicting demand-critical entries too aggressively. |
| `PILOT_REAL_FORCE=1` | off | research, Qwen only | Force real pilot behavior despite a cache too small for the normal guard. |
| `COUPLE=PATH` | unset | research | Load cross-layer expert-coupling predictions. |
| `COUPLE_K=N` | `8` | research | Number of coupling predictions, clamped to `1..32`. |
| `COUPLE_D=N` | `1` | research | Coupling depth, clamped to `1..2`. |

Granite and Qwen write cumulative usage to `MODEL.coli_usage`; their direct parsers do not
currently expose `--usage-file`. To reuse a sidecar stored elsewhere, place or symlink it at the
expected adjacent path. GPT-OSS uses the same default naming but can override it with
`--usage-file` or `COLI_USAGE_FILE`.

When `PIN` is not explicitly set, `AUTOPIN=1` begins automatic hot pinning only after at least
5,000 cumulative expert selections. Before that threshold, the scheduler uses its ordinary cache
without trustworthy automatic pins.

### Qwen3-Next-only variables

| Variable | Default | Meaning |
|---|---:|---|
| `QWEN_ROUTER_GPU=0|1` | `1` | Run router selection on CUDA. On GTX 1060, `0` was measured faster. |
| `QWEN_FOCUS_LAYER_COUNT=K` | `0` | Concentrate hot pins in K history-selected layers. |
| `QWEN_FOCUS_GROUP=0|1` | on when focus count >0 | Group already-resident experts in focused layers. |
| `QWEN_FOCUS_GROUP_MIN=N` | `2` | Minimum resident hits required for a grouped launch. |

### Qwen3.5/Qwen3.6-only variables

| Variable | Default | Meaning |
|---|---:|---|
| `Q35_ROUTER_GPU=0|1` | `0` | CUDA router selection; CPU is default. |
| `Q35_FOCUS_LAYER_COUNT=K` | `0` | Concentrate hot pins in K selected trunk layers. |
| `Q35_FOCUS_GROUP=0|1` | on when focus count >0 | Group resident experts in focused layers. |
| `Q35_FOCUS_GROUP_MIN=N` | `2` | Minimum grouped experts. |
| `QWEN35_MTP_DRAFT=0..8` | model-dependent/CLI value | Environment alternative to `--mtp-draft`. |
| `Q35_EXPERT_ZERO_THRESHOLD=X` | `0` | Research/quality-changing threshold for treating very small expert values as zero. |

### GPT-OSS-only variables

| Variable | Default | Status | Meaning |
|---|---:|---|---|
| `COLI_USAGE_FILE=PATH` | `MODEL.coli_usage` | normal | Override usage-history file. |
| `GPTOSS_EXPERT_CACHE_MODE=stats|fixed` | `stats` | normal | Environment cache mode when CLI does not specify one. |
| `GPTOSS_EXPERT_CACHE_PER_LAYER=N` | unset | normal | Backward-compatible fixed LRU size; selects fixed mode. |
| `GPTOSS_EXPERT_CACHE_RESERVE_MIB=N` | `384` | normal | Keep this much free VRAM; `CUDA_RESERVE_GB` can raise the reserve. |
| `GPTOSS_EXPERT_CACHE_MIB=N` | all available after reserve | normal | Cap global expert-cache budget. |
| `GPTOSS_MMAP_READERS=N` | off | normal/measure | Parallel copy workers for selected mmap-backed gate/up/down fragments; capped at `3 × top_k`. |
| `GPTOSS_TRACE_STARTUP=1` | off | diagnostic | First-forward timing, faults, I/O, and RSS trace. |
| `GPTOSS_TRACE_TENSORS=1` | off | diagnostic | Add per-tensor residency details to startup trace. |
| `COLI_DEBUG_LIGHT=1` | set by `--debug-light` | diagnostic | Reduced GPT-OSS debug output/progress behavior. |

In stats mode, only experts with recorded hits (plus bundles already resident through static
placement) are selected. With an empty usage file, the startup log explicitly reports that no
history is available. `GPTOSS_MMAP_READERS` performs parallel **demand materialization after
routing**; it does not prefetch a future layer and does not overlap next-layer prediction with
current-layer computation.

### Focus-history helper

```bash
python3 c/tools/qwen_focus_tool.py recommend \
  "$MODEL.coli_usage" --pin-slots 73 --max-layers 48
```

Benchmark selected focus counts with the helper's `benchmark` subcommand. Its model is an
estimate; measured tok/s is authoritative.

---

## 6. SGGUF sparse conversion

### Capability

Only eligible 3-D tensors named `blk.N.ffn_gate_exps.weight`,
`blk.N.ffn_up_exps.weight`, or `blk.N.ffn_down_exps.weight` are sparsified. Attention, router,
norms, embeddings, output, biases, and ineligible expert tensors are copied byte-for-byte. The
input must be a standard GGUF. The current sparse format is `SPB3`: 256-weight groups, an
occupancy bitmap, codec metadata, and retained values/codes.

### Build

```bash
cd ~/GitHub/colibri/c
make sgguf-convert sgguf-inspect -j"$(nproc)"
```

### Convert

```bash
./c/sgguf-convert INPUT.gguf OUTPUT.sgguf \
  --threshold 0.01 \
  --codec auto \
  --jobs auto \
  --max-output-gb 120
```

### Converter flags

| Flag | Values | Default/meaning |
|---|---|---|
| `--threshold N` | float | Absolute sparsity threshold; default `0.01`. This changes model values. |
| `--codec auto` | — | Exact source-codec storage for supported codecs; otherwise retained-F16 fallback. |
| `--codec preserve` | — | Current alias of `auto`. |
| `--codec f16|bf16|f32` | — | Force retained values into the selected scalar type. |
| `--jobs N`, `-j N`, `--mp N` | integer | Worker count; default `1`. |
| `--jobs auto`, `--mp auto` | — | Automatic multiprocessing. |
| `--max-output-gb N` | float | Optional output ceiling; `0`/omitted means unlimited. |
| `--no-verify` | switch | Disable semantic sample verification before final rename. Verification is on by default. |
| `--quiet` | switch | Reduce output. |

Run it exactly like GGUF afterward:

```bash
./c/colibri --gguf OUTPUT.sgguf --device cuda --prompt 'Hello' --max-tokens 40
```

Each sparse tensor's relative-offset stream is limited to less than 4 GiB; this is a
per-tensor limit, not a whole-file limit.

---

## 7. Inspectors

### General GGUF inspector

```bash
./c/gguf-inspect --metadata --tensors MODEL.gguf
./c/gguf-inspect --find-tensor 'blk.0.ffn_gate_exps.weight' MODEL.gguf
```

Flags: `--metadata`, `--tensors`, `--find-tensor NAME`.

### GGUF/SGGUF storage inspector

```bash
./c/sgguf-inspect MODEL.gguf
./c/sgguf-inspect MODEL.sgguf --tensors
```

Flags: `--tensors`, `--no-sparse-details`.

The SGGUF inspector reports architecture, tensor types, routed-expert bytes, unsupported
types, and sparse accounting. It exits nonzero for unsupported model tensor types.

---

## 8. GPT-OSS three-file split storage

The current hot split planner is architecture-specific: `split_plan.c` requires
`gpt-oss.block_count` and `gpt-oss.expert_count`. Do not present this workflow as verified for
Granite or Qwen models without extending and testing the planner.

### Layout

| File | Role |
|---|---|
| `*.fast.split.gguf` | Primary metadata and fast-disk tensor payload, mmap-backed. |
| `*.preload.shard` | Tensors loaded once from slow storage into RAM or VRAM. |
| `*.tail.shard` | Remaining slow-disk tensors, mmap-backed on demand. |

The primary embeds absolute paths to the other two files. Run only the primary path.
Moving a shard afterward breaks the embedded path unless the split is rebuilt.

### Planner

```bash
./c/sgguf-plan "$MODEL" \
  --fast-dir /fast/disk \
  --slow-dir /slow/disk \
  --usage-file "$MODEL.coli_usage" \
  --context 4096 \
  --fast-reserve-mib 384 \
  --ram-breathing-mib 384 \
  --vram-breathing-mib 128 \
  --vram-workspace-mib 256
```

Planner flags:

| Flag | Default | Meaning |
|---|---:|---|
| `--fast-dir DIR` | required | Fast-storage destination. |
| `--slow-dir DIR` | required | Slow-storage destination. |
| `--usage-file PATH` | required | `layer expert cumulative_selection_count`. |
| `--prefix NAME` | derived | Output basename. |
| `--fast-reserve-mib N` | `384` | Free room left on fast disk. |
| `--ram-breathing-mib N` | `384` | RAM left after runtime allocations. |
| `--ram-reserve-mib N` | alias | Compatibility alias for RAM breathing room. |
| `--vram-breathing-mib N` | `128` | VRAM left after runtime allocations. |
| `--vram-reserve-mib N` | alias | Compatibility alias for VRAM breathing room. |
| `--context N` | `4096` | Context used for KV/runtime estimates. |
| `--vram-workspace-mib N` | `256` | CUDA context/scratch allowance. |
| `--runtime-ram-mib N` | automatic | Override runtime RAM estimate. |
| `--runtime-vram-mib N` | automatic | Override runtime VRAM estimate. In stats-cache planning, automatic accounting reserves workspace; the dynamic stats cache consumes VRAM left after static placement. |
| `--device N` | `0` | NVIDIA GPU queried with `nvidia-smi`. |
| `--placements-output FILE` | unset | Write every tensor placement as TSV. |
| `--command-only` | off | Print only the recommended split command. |

The planner uses binary MiB/GiB and current `MemAvailable`, free disk, and free VRAM.
It intentionally does not use ambiguous legacy `CUDA_RESERVE_GB` for split planning.

### Splitter

Usually run the exact command printed by the planner. Manual form:

```bash
./c/sgguf-split "$MODEL" \
  --usage-file "$MODEL.coli_usage" \
  --fast-output /fast/model.fast.split.gguf \
  --preload-output /slow/model.preload.shard \
  --tail-output /slow/model.tail.shard \
  --fast-bytes 100GiB \
  --ram-bytes 12GiB \
  --vram-bytes 4GiB \
  --verify
```

`SIZE` accepts bytes or `K`, `M`, `G`, `T`, optionally `iB`/`B`, for example
`4000000000`, `4G`, or `4GiB`.

Splitter switches: `--dry-run`, `--verify`, `--overwrite`, `--verbose`/`-v`.

Run the resulting model:

```bash
COLI_SPLIT_VERBOSE=1 \
./c/colibri --gguf /fast/model.fast.split.gguf --device cuda \
  --usage-file "$MODEL.coli_usage" \
  --prompt 'Hello' --max-tokens 60 --verbose
```

Split-loader diagnostics:

| Variable | Meaning |
|---|---|
| `COLI_SPLIT_VERBOSE=1` | Print detailed logical-to-physical tensor mapping/preload activity. |
| `COLI_RAM_PROGRESS=1` | Print RAM preload progress. |
| `COLI_DEBUG_LIGHT=1` | Reduce heavy progress/debug behavior in supported paths. |

---

## 9. `coli` launcher: GLM snapshot runtime

This section is for a converted **directory snapshot**, not a GGUF file.

### Subcommands

| Command | Purpose |
|---|---|
| `./c/coli build` | Build/prepare the engine. |
| `./c/coli info` | Model, build, RAM, disk, and configuration status. |
| `./c/coli plan` | Compute RAM/VRAM/disk placement plan. |
| `./c/coli doctor` | Installation and execution-plan diagnostics. |
| `./c/coli run "prompt"` | One-shot generation. |
| `./c/coli chat` | Interactive chat; may attach to an existing server. |
| `./c/coli serve` | OpenAI-compatible persistent API server. |
| `./c/coli stop` | Shut down a local server. |
| `./c/coli web` | Start server and dashboard. |
| `./c/coli bench [tasks...]` | HellaSwag/ARC/MMLU-style evaluation. |
| `./c/coli convert` | Download/convert GLM FP8 snapshot to Colibri storage. |

### Common `coli` flags

| Flag | Default | Maps to/effect |
|---|---:|---|
| `--model DIR` | `$COLI_MODEL` or built-in path | Model snapshot; engine receives `SNAP`. |
| `--ram N` | `0` auto | `RAM_GB`, expert working-set budget in GB. |
| `--auto-tier` | off | Apply computed RAM/VRAM plan automatically. |
| `--ctx N` | `0` auto | `CTX`. |
| `--gpu auto|none|0,1` | unset | GPU selection/hard CPU off-switch. |
| `--vram N` | `0` auto | Total VRAM planning budget in GB. |
| `--policy quality|balanced|experimental-fast` | `quality` | `COLI_POLICY`. |
| `--repin N` | `0` | `REPIN`. |
| `--cap N` | `8` | Initial expert-cache slots per layer. |
| `--ngen N` | `1024` | `NGEN`. |
| `--topp P` | `0` | `TOPP`, expert/router reduction in this engine, with quality warning. |
| `--topk N` | `0` | `TOPK`, expert/router reduction in this engine, with quality warning. |
| `--temp X` | engine default | `COLI_TEMP`; `0` is greedy. |

### `chat`

| Flag | Meaning |
|---|---|
| `--attach [URL]` | Use an already-running server; bare form probes `http://127.0.0.1:8000`. |
| `--no-attach` | Always spawn a private engine. |
| `--api-key KEY` | Bearer key for attached server. |

### `serve` and `web`

| Flag | Default | Meaning |
|---|---:|---|
| `--host HOST` | `127.0.0.1` | Bind address. |
| `--port N` | `8000` | Port. |
| `--model-id ID` | `glm-5.2-colibri` | API model identifier. |
| `--api-key KEY` | unset | Require bearer authentication. |
| `--cors-origin ORIGIN` | unset, repeatable | Allowed CORS origin. |
| `--max-queue N` | `8` | Queued requests. |
| `--queue-timeout SEC` | `300` | Queue wait timeout. |
| `--kv-slots N` | `1` | Independent conversation KV slots. |
| `web --no-browser` | off | Do not open browser automatically. |

`stop` supports `--port N` and `--dry-run`.

### `bench`

```bash
./c/coli bench hellaswag arc_challenge mmlu --limit 40 --data ~/.cache/colibri/bench
```

### `convert`

| Flag | Default | Meaning |
|---|---:|---|
| `--repo` | `zai-org/GLM-5.2-FP8` | Source repository. |
| `--ebits N` | `4` | Routed-expert bit width. |
| `--io-bits N` | `8` | Resident/dense bit width. |
| `--xbits N` | `0` | Extra override bit width. |
| `--no-mtp` | off | Skip separate MTP conversion. |

The converter normally performs the main model conversion, then converts the MTP head at
at least int8 because int4 MTP had very low measured acceptance.

### Download/conversion environment variables

| Variable | Default | Scope | Meaning |
|---|---:|---|---|
| `COLI_DL_STREAMS=N` | `2` | FP8 conversion downloader | Segmented download streams, clamped to `1..8`; set `1` for single-stream/resume compatibility. |
| `HF_TOKEN=...` | unset | Hugging Face downloads | Authorization token for gated/private files. |
| `XDG_CACHE_HOME` | `~/.cache` | Linux launcher | Base cache directory used by `coli`. |
| `LOCALAPPDATA` | Windows default | Windows launcher | Base cache directory used by `coli`. |
| `GLM_REVISION`, `GLM_HF_REVISION`, `GLM_MS_REVISION` | repository defaults | standalone download helpers | Optional revision/commit pins for supply-chain reproducibility; these are helper-script settings, not normal inference flags. |

---

## 10. GLM snapshot engine: normal settings

These variables apply to the directory/snapshot runtime. They do not automatically change
direct GGUF behavior unless the GGUF model-specific code explicitly reads the same name.

### Generation and state

| Variable | Default | Meaning |
|---|---:|---|
| `SNAP=DIR` | required in low-level mode | Snapshot directory. |
| `RAM_GB=X` | auto, roughly 88% available | RAM expert-cache budget. |
| `CTX=N` | `4096` | KV context allocation. |
| `NGEN=N` | `64` low-level; launcher uses `1024` | Maximum generated tokens. |
| `COLI_TEMP=X` | auto | Token temperature; `0` greedy. Prefer this over deprecated `TEMP`. |
| `NUCLEUS=P` | `0.90` | Token-sampling nucleus mass. |
| `SEED=N` | clock/PID | Sampling seed. |
| `KVSAVE=0|1` | `1` | Save/load `.coli_kv` conversation state. |
| `KV_SLOTS=N` | `1` | Independent serve-mode KV slots. |
| `THINK=0|1` | `0` | Visible `<think>` output in the engine path. |
| `CHAT_TEMPLATE=0|1` | `1` | Apply GLM chat formatting; `0` is raw. |
| `GRAMMAR=FILE` | unset | GBNF constraint. |
| `SCHEMA=FILE` | unset | JSON Schema compiled to GBNF when `GRAMMAR` is absent. |
| `GRAMMAR_DRAFT=N` | unset | Maximum grammar-forced speculative draft span. |

### Expert residency and learning cache

| Variable | Default | Meaning |
|---|---:|---|
| `PIN=auto|PATH` | unset | Load permanent hot experts from history/stats. |
| `PIN_GB=X|all` | `10` when explicit low-level `PIN` | Pinned hot-store size. |
| `AUTOPIN=0|1` | `1` | Automatically pin as `.coli_usage` becomes trustworthy. |
| `PIN_FILL=0|1` | `0` | Fill remaining pin capacity without measured heat. |
| `CAP_RAISE=0|1` | `1` | Raise cache cap above top-k when RAM permits. |
| `REPIN=N` | `0` | Periodically move hot experts between tiers. |
| `REPIN_VERBOSE=1` | off | Print individual repin swaps. |
| `RSS_GUARD_GB=X` | RAM budget | Explicit resident-memory ceiling during repin/pilot. |
| `COLI_RAM_OVERCOMMIT=1` | off | Bypass projected peak-vs-MemAvailable safety exit. Risk of OOM kill. |

### Disk and I/O

| Variable | Default | Meaning |
|---|---:|---|
| `PIPE=0|1` | Linux off, Windows on | Overlap expert reads and computation. |
| `PIPE_WORKERS=N` | `8` | Loader/io-wq worker count; setting it implies pipe unless `PIPE=0`. |
| `COLI_PIPE_BLOCK=1` | off | Use blocking wait instead of spin wait in pipe. |
| `DIRECT=0|1` | `0` | O_DIRECT/unbuffered expert reads; benchmark per drive. |
| `URING=0|1` | `0` | Linux io_uring queued reads; implies pipe; incompatible with `COLI_MMAP=1`. |
| `COLI_MMAP=0|1` | `0` | mmap expert slabs and use page cache as cache. |
| `PREFETCH=N` | `0` | Legacy streamed-expert prefetch depth. |
| `COLI_MODEL_MIRROR=DIR` | unset | Second byte-identical/partial model copy on another drive. |
| `SNAP_MIRROR=DIR` | unset | Compatibility alias for mirror directory. |
| `COLI_DISK_WEIGHTS=A,B` | measured | Primary/mirror bandwidth ratio. |
| `COLI_MODEL_DIRS=...` | unset | Multi-directory shard lookup used by storage code. |
| `COLI_DISKCLASS_WINDOW=N` | calibrated | Override hot/cold disk classification recency window. |
| `DISK_SPLIT=1` | off | Split disk timing by execution phase in reports. |

### CPU and numerical kernels

| Variable | Default | Status | Meaning |
|---|---:|---|---|
| `COLI_NO_OMP_TUNE=1` | off | normal escape hatch | Disable one-time hot-thread OpenMP re-exec/tuning. |
| `COLI_NUMA=1` | off unless planned | normal | Interleave large slabs on multi-socket Linux. |
| `MLOCK=-1|0|1` | auto | normal | Wire resident memory; auto mainly on macOS. |
| `IDOT=0|1` | `1` | normal | Integer dot kernel; `0` selects exact float A/B path. |
| `I4S=N` | source threshold | research | Use int4 IDOT only for batches with `S>=N`. |
| `I4_ACC512=1` | off | research | AVX-512 int4 accumulator path. |
| `I4_ACC512_TEST=1` | off | test | Run self-test and exit. |
| `COLI_NO_FUSED_PAIR=1` | off | A/B | Disable fused-pair kernel. |
| `XEXP=1` | off | experimental | One parallel region across expert block in eligible S=1 path. |
| `ABSORB=-1|0|1` | `-1` auto | A/B | MLA absorption: auto for small S, never, or always. |
| `NOPACK=1` | off | debug/A-B | Disable weight packing. |
| `DROP=1` | off | internal | Drop-path experiment/debug switch. |

### GLM speculative decoding

| Variable | Default | Meaning |
|---|---:|---|
| `MTP=0|1` | model-enabled | Master MTP availability toggle during model load. |
| `SPEC=0|1` | `1` | Speculative decoding switch. |
| `DRAFT=N` | `-1` auto | Number of draft tokens; auto resolves after model load. |
| `SPEC_PIN=0|1` | `1` | Keep draft and verification in the same kernel family. |
| `COLI_CUDA_MTP=1` | off | Opt into MTP under CUDA; valuable mainly near full expert residency. |
| `MTP_DEBUG=1` | off | MTP diagnostics. |
| `MTP_PRENORM=1` | off | MTP ablation. |
| `MTP_SWAP=1` | off | MTP ablation. |

### Router/prefetch experiments

| Variable | Default | Status | Meaning |
|---|---:|---|---|
| `PILOT=1` | off | experimental | Router-lookahead prefetch. |
| `PILOT_REAL=1` | off | experimental | Actual cross-layer expert loads. |
| `PILOT_TWO=1` | off | experimental | Two-step shared-expert-corrected prediction. |
| `PILOT_K=N` | `8`, real=`6` | experimental | Prediction width. |
| `PILOT_EVICT_GUARD=0|1` | `1` | safeguard | Protect demand cache from speculative eviction. |
| `LOOKA=1` | off | telemetry | Measure several routing predictors without making them policy. |
| `COUPLE=FILE` | unset | experimental | Cross-layer coupling model. |
| `COUPLE_K=N` | `8` | experimental | Coupling prediction count. |
| `COUPLE_D=1|2` | `1` | experimental | Coupling depth. |
| `CACHE_ROUTE=1` | off | quality-changing experiment | Prefer resident experts within top-M/mass window. |
| `ROUTE_J=N` | `2` | experiment | Highest ranks that may never be substituted. |
| `ROUTE_M=N` | `12` | experiment | Candidate rank window. |
| `ROUTE_P=P` | `0` | experiment | Use cumulative router mass instead of fixed M. |
| `ROUTE_ALPHA=X` | `1` | experiment | Down-weight substituted expert gate mass. |
| `ROUTE_AGREE=1` | auto with CACHE_ROUTE | telemetry | Report overlap and KL versus true top-k. |
| `ROUTE_TRACE=FILE` | unset | telemetry | Record per-position, per-layer routing. |
| `TOPK=N` | `0` | quality-changing | Reduce selected expert count. |
| `TOPP=P` | `0` | quality-changing | Drop low-weight routed experts by cumulative mass. |
| `EXPERT_BUDGET=N` | forced off | quarantined | Broken operating window; requires explicit experimental unlock. |
| `EXPERT_BUDGET_EXPERIMENTAL=1` | unset | dangerous research | Enables `EXPERT_BUDGET`; source warns to expect incoherent output. |

### DSA/attention experiments

| Variable | Default | Meaning |
|---|---:|---|
| `DSA=0|1` | on/model-driven | Dynamic Sparse Attention indexer. |
| `DSA_FORCE=1` | off | Force DSA path for validation. |
| `DSA_TOPK=N` | model value | Override indexer top-k for tests. |

### GLM CUDA tier and pipeline

| Variable | Default | Meaning |
|---|---:|---|
| `COLI_CUDA=1` | off | Enable CUDA backend; requires CUDA build. |
| `COLI_GPU=N` | device 0 | Select one device. |
| `COLI_GPUS=0,1,...` | unset | Select multiple devices; do not set with `COLI_GPU`. |
| `CUDA_DENSE=1` | off | Put dense non-expert matrices on GPU. |
| `CUDA_EXPERT_GB=X|auto` | `0` | VRAM expert tier budget. |
| `CUDA_RESERVE_GB=X` | `2.0` in GLM engine | VRAM protected from expert tier. |
| `CUDA_RELEASE_HOST=0|1` | on for multi-GPU | Drop host backing after upload. |
| `COLI_CUDA_ATTN=1` | off | Small-batch attention on GPU. |
| `COLI_CUDA_ATTN_SHARD=1` | off | Shard KV heads across GPUs. |
| `COLI_CUDA_PIPE=1|2` | `0` | GPU-resident pipeline modes. |
| `COLI_CUDA_PIPE_SHARD=1` | off | Multi-device P2P head-shard attention path. |
| `COLI_CUDA_PIPE_S_MIN=N` | single GPU `1`, multi `8` | Minimum batch S for pipe2. |
| `COLI_CUDA_ROUTER=1` | off | GLM router on layer's home GPU. |
| `COLI_CUDA_RESID=1` | off | Keep grouped expert results resident on GPU. |
| `COLI_GROUP_ASYNC=1` | off | Asynchronous grouped-expert launch experiment. |
| `COLI_CUDA_PROFILE=1` | off | CUDA timing statistics. |
| `COLI_CUDA_ASYNC=0` | on by default | Disable async copies/pinned staging. |
| `COLI_CUDA_DUAL_PROJ=0` | on by default | Split fused gate+up launch. |
| `COLI_CUDA_W4_PACKED=0` | on by default | Disable packed W4 grouped path. |
| `COLI_CUDA_TC_INT4=1` | off | W4A4 Tensor Core path. |
| `COLI_CUDA_TC_MIN_ROWS=N` | `8` | W4A4 row threshold. |
| `COLI_CUDA_TC_W4A16=1` | off | Lossless W4A16 Tensor Core path. |
| `COLI_CUDA_TC_W4A16_MIN=N` | `16` | W4A16 row threshold. |
| `COLI_CUDA_SHARED_W4A16=1` | off | Shared-expert W4A16 Tensor Core path. |
| `COLI_CUDA_SHARED_W4A16_MIN_ROWS=N` | `32` | Shared-expert row threshold. |

### Metal

| Variable | Default | Meaning |
|---|---:|---|
| `COLI_METAL=1` | off | Enable Metal backend in a Metal build. |
| `COLI_METAL_GEMM_MIN=N` | `16` | Minimum matmul rows sent to GPU. |
| `COLI_METAL_SPIN=1` | off | Keep-alive spinner; latency vs power trade-off. |
| `COLI_METAL_UNTRACKED=1` | off | Untracked Metal resource hazard mode. |
| `COLI_METAL_RESSET=...` | internal | Metal residency-set experiment/internal control. |

---

## 11. Server and API capabilities

`coli serve` launches `c/openai_server.py`, which provides an OpenAI-compatible
chat-completions API and tool-call handling. The server keeps the engine and caches warm.

| Variable | Default | Meaning |
|---|---:|---|
| `COLI_MODEL=DIR` | unset | Default snapshot. |
| `COLI_MODEL_ID=ID` | `glm-5.2-colibri` | Reported API model name. |
| `COLI_API_KEY=KEY` | unset | Bearer authentication. |
| `COLI_MAX_QUEUE=N` | `8` | Queue capacity. |
| `COLI_QUEUE_TIMEOUT=SEC` | `300` | Queue timeout. |
| `COLI_KV_SLOTS=N` | `1` | Conversation slots. |
| `COLI_ALLOW_INSECURE_BIND=1` | off | Permit unsafe public bind behavior without normal protection. |
| `COLI_DEBUG=1` | off | Tee model output stream to server stderr. |
| `COLI_DEBUG=2` | off | Tee rendered prompt and output, correlated by request. |
| `COLI_TOOL_SALVAGE=1` | off | Repair a narrow class of malformed int4 tool calls. |
| `COLI_THINK=1` | off | Default thinking behavior when client provides no explicit choice. |
| `COLI_RAW=1` | off | Raw launcher output mode. |
| `COLI_COLOR=1` | TTY auto | Force colored launcher output. |
| `COLI_ENGINE=PATH` | auto-discovered | Override native engine location. |

Low-level engine mode switches `SERVE`, `SERVE_BATCH`, `SCORE`, `PROMPT`, and internal
sentinel `COLI_OMP_TUNED` are normally set by the launcher and should not be hand-managed.

---

## 12. Diagnostics, telemetry, and validation modes

| Setting/tool | Purpose |
|---|---|
| `--verbose` | Primary direct-GGUF telemetry: model geometry, scheduler, CUDA, MTP, tok/s. |
| GPT-OSS `--debug` | Structured token/layer/expert operation logs. |
| `GPTOSS_TRACE_STARTUP=1` | Startup/first-forward timing and fault trace. |
| `PROF=1` | GLM performance profile and bottleneck verdict. |
| `STATS=FILE` | Write final expert usage histogram. |
| `TOKENS=1` | Dump generated token IDs for A/B checks. |
| `ROUTE_TRACE=FILE` | Dump router selections/gates. |
| `DISK_SPLIT=1` | Break disk timing down by forward/draft/absorb context. |
| `SCORE=REQUESTS` | Low-level log-likelihood scoring mode. |
| `SCORE_PREFIX=0|1` | Control GLM scoring prefix. |
| `TF=1` | Teacher-forcing oracle comparison. |
| `REF=FILE` | Oracle JSON path. |
| `REF_FORCE=1` | Run mismatched oracle anyway. |
| `REPLAY=1` | Replay reference token sequence. |
| `I4_ACC512_TEST=1` | Kernel self-test and exit. |
| `make test` | Complete regular test suite. |
| `make cuda-test ...` | CUDA backend correctness binary. |
| `make cuda-sgguf-test ...` | Sparse CUDA correctness test. |

`COLI_EFFICIENCY_*`, `COLI_GPU_FAIL_AFTER`, `COLI_PROFILE_SUM_TOL`,
`COLI_MAX_DISK_WAIT_SHARE`, `COLI_MIN_CPU_CUDA_AGREE`, `COLI_TINY_TOK_S_FLOOR`,
and similar names in tests are harness controls, not normal runtime settings.

---

## 13. Separate OLMoE sister engine

`c/olmoe.c` is a separate executable and has its own research controls such as
`HOT`, `WARMUP`, `SMOOTH`, `CONF_LIMIT`, `WIDE`, and `PPL`. Do not assume those
variables affect `c/colibri`; they are intentionally excluded from the main recipes.

Build it with:

```bash
cd ~/GitHub/colibri/c
make olmoe -j"$(nproc)"
```

---

## 14. Stability classification

Use this classification when cleaning the project:

### Safe normal surface

- CPU/CUDA build controls.
- Direct GGUF common flags and the verified Qwen3.5/GPT-OSS repetition penalty.
- `PIN`, `AUTOPIN`, `CUDA_RESERVE_GB`, `CUDA_EXPERT_GB`.
- GPT-OSS stats/fixed cache controls.
- Inspect/convert/split utilities with verification enabled.
- `coli run`, `chat`, `serve`, `web`, `plan`, `doctor`.

### Measure before keeping

- `DIRECT`, `PIPE`, `URING`, `GPTOSS_MMAP_READERS`.
- GPU router toggles.
- Qwen MTP draft count.
- Focused pin/grouping.
- CUDA Tensor Core opt-ins and pipeline modes.

### Research/quality-changing

- `PILOT_REAL`, `PILOT_TWO`, coupling, live repin.
- `CACHE_ROUTE`, router `TOPK`/`TOPP` reduction.
- sparse conversion threshold changes.
- DSA/kernel A/B controls.

### Quarantined or internal

- `EXPERT_BUDGET` unless deliberately testing known-bad behavior.
- MTP ablations, `DROP`, `NOPACK`, forced failure and efficiency-test variables.
- internal launch-mode sentinels.

---

## 15. Known traps

1. **`coli` and direct GGUF are different interfaces.** A flag existing in one does not
   mean it exists in the other.
2. **Direct GGUF is one-shot and greedy.** `--repeat-penalty` modifies logits but does not add
   temperature, top-p, or random sampling.
3. **Repeat penalty changes routing history.** Different output tokens produce different MoE
   expert selections. Compare cache or I/O performance with the same prompt, penalty, usage file,
   and initial cache state.
4. **Qwen usage files are fixed sidecars.** Expect `MODEL.coli_usage`; Qwen direct runtimes do not
   currently provide `--usage-file`. Use a symlink when the sidecar lives elsewhere.
5. **Automatic Qwen/Granite pins need history.** `AUTOPIN=1` waits for 5,000 cumulative selections.
6. **GPT-OSS stats mode can start empty.** With no recorded hits, use fixed mode to collect a
   representative usage file before evaluating stats-greedy placement.
7. **Qwen router defaults differ.** Qwen3-Next defaults GPU router on; Qwen3.5/3.6 defaults it off.
8. **Qwen MTP acceptance is not speed.** On GTX 1060, draft 0 was faster in the documented test.
9. **Qwen3.5 and GPT-OSS echo formatted input.** Their stdout is input protocol plus generated
   output, not generated text alone.
10. **GPT-OSS Harmony `<|end|>` is not a generation stop.** It separates messages; EOS,
    `<|return|>`, and `<|call|>` terminate generation.
11. **GPT-OSS stats and fixed cache are different policies.** Do not combine
    `--expert-cache-mode stats` with `--expert-cache-per-layer`.
12. **`GPTOSS_MMAP_READERS` is demand reading, not future-layer prefetch.** It parallelizes
    selected gate/up/down mmap copies after routing.
13. **The current three-file hot split is GPT-OSS-specific.** The planner requires GPT-OSS
    block/expert metadata.
14. **Split files contain absolute paths.** Decide final locations before splitting.
15. **Keep conversion verification on.** Use `--no-verify` only for controlled experiments.
16. **Do not use `TEMP` casually.** Prefer `COLI_TEMP`; `TEMP` is commonly a temporary-directory
    variable on Windows/ROCm environments.
17. **`URING=1` and `COLI_MMAP=1` are incompatible.**
18. **A CUDA build is not CUDA execution.** Direct GGUF requires `--device cuda`; GLM mode
    requires launcher GPU planning or `COLI_CUDA=1`.
19. **The repository still contains older naming in prose.** `glm` is a compatibility target;
    the actual built engine is now `c/colibri`.

---

## 16. Fast source-of-truth commands

When code changes, these commands reveal the actual current surface.

```bash
# coli flags
./c/coli --help
./c/coli serve --help
./c/coli convert --help

# direct GGUF parsers
rg -n 'Usage:|--[a-z][a-z0-9-]+' \
  c/gguf_granite.c c/gguf_qwen3next.c c/gguf_qwen35moe.c c/gguf_gptoss.c

# environment-variable readers (C and Python)
rg -n 'getenv\("[A-Z0-9_]+|os\.(environ\.get|getenv)\(' c

# verify recently added direct-runtime features
rg -n 'repeat-penalty|GPTOSS_MMAP_READERS|expert-cache-mode|Echo the exact formatted|Harmony control' c

# build targets and build variables
make -C c -pn | less

# utility help
./c/gguf-inspect 2>&1
./c/sgguf-convert 2>&1
./c/sgguf-inspect 2>&1
./c/sgguf-plan 2>&1
./c/sgguf-split 2>&1
```

### Verification notes for this revision

The following material corrections were made during this audit:

- Qwen3.5/Qwen3.6 now documents its implemented `--repeat-penalty` and formatted-input echo.
- GPT-OSS Harmony separators, stop-token behavior, dual cache modes, and demand-reader scope are explicit.
- The three-file hot-split workflow is correctly marked GPT-OSS-specific.
- SGGUF eligibility, codec fallback, defaults, and verification behavior are stated precisely.
- Qwen/Granite sidecar and 5,000-selection `AUTOPIN` behavior are documented.
- Benchmark guidance now warns that repetition penalty changes expert routing statistics.

The code is authoritative when this document and source diverge.
