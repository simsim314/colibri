# Three-file split storage

The split loader presents three physical files as one logical GGUF/SGGUF model:

- `*.fast.split.gguf`: sparse primary on the fast disk, mmap-backed;
- `*.preload.shard`: tensors read once from the slow disk into RAM or VRAM;
- `*.tail.shard`: remaining slow-disk tensors, mmap-backed on demand.

The fast primary embeds absolute paths and a tensor-location manifest. Pass only
the fast-primary path to `colibri`.

## Planner v2

`sgguf-plan` inspects current resources and prints explicit budgets for
`sgguf-split`. Current processes are already accounted for through Linux
`MemAvailable` and `nvidia-smi` free VRAM.

The planner uses binary units deliberately:

- 1 MiB = 1,048,576 bytes;
- 1 GiB = 1,073,741,824 bytes.

It reserves separately:

- fast-disk breathing room;
- runtime RAM, including GPT-OSS CPU KV cache for the selected context;
- RAM breathing room;
- the per-layer GPT-OSS dynamic expert cache in VRAM;
- CUDA context and temporary workspace;
- VRAM breathing room for the desktop/browser and allocation jitter.

`CUDA_RESERVE_GB` is not used by planner v2. It was too ambiguous because it
mixed permanent tensors, dynamic expert cache, CUDA scratch and free headroom.
Use the explicit planner flags instead.

For GPT-OSS, dense q/k/v/o/router residency is currently all-or-none: if one
matrix fails to reside, the runtime releases the whole dense set and streams it.
The planner therefore recommends either the complete dense set or zero partial
dense preload. Expert weights remain in the dynamic per-layer cache. Without
statistics, this version does not permanently select individual hot experts.

Example:

```bash
./c/sgguf-plan MODEL.gguf \
  --fast-dir /home/simsim314/Downloads \
  --slow-dir /mnt/pacer \
  --fast-reserve-mib 384 \
  --ram-breathing-mib 384 \
  --vram-breathing-mib 128 \
  --vram-workspace-mib 256 \
  --context 4096 \
  --expert-cache-per-layer 4
```

The default output is concise. To save every tensor decision without flooding
the terminal:

```bash
./c/sgguf-plan MODEL.gguf ... \
  --placements-output /tmp/model-split-placement.tsv
```

Run the exact `sgguf-split` command printed by the planner. Manual budgets are
also supported directly by `sgguf-split`.

At runtime, the GPT-OSS CUDA cache is selected automatically from the usage
file and the VRAM that remains after split preloads and dense weights:

```bash
GPTOSS_EXPERT_CACHE_RESERVE_MIB=384 \
COLI_SPLIT_VERBOSE=1 \
./c/colibri --gguf MODEL.fast.split.gguf --device cuda \
  --usage-file MODEL.coli_usage ...
```

`GPTOSS_EXPERT_CACHE_MIB` can cap the automatic runtime cache. Cache capacity is
allocated globally by historical selections per encoded byte, not by a fixed
number of experts per layer. `COLI_SPLIT_VERBOSE=1` reports actual RAM and VRAM
preload activity.
