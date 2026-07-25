#!/usr/bin/env python3
"""Simulate the compact, directly indexed SGGUF layout.

The simulator scans EVERY tensor before touching expert weights and creates one
model-global type dictionary in the SGGUF header.

For U distinct GGML tensor types:

    compact_type_bits = ceil(log2(U))

Each tensor descriptor uses exactly that many bits. The header table maps those
compact IDs to the raw GGML type IDs, themselves packed with only the bit width
actually required by this model.

Routed MoE experts use 256 logical positions per occupancy group. There are no
fixed 64-block pages. Retained values are packed in units of exactly 8 codes:

    Q6_K -> 8 x 6 bits = 6 bytes
    Q8_0 -> 8 x 8 bits = 8 bytes
    Q5_K -> 8 x 5 bits = 5 bytes

Three occupancy encodings are compared while keeping the same retained values
and quantization auxiliary streams:

    tree
        preorder 0/10/11 tree for every 256-value group

    bitmap
        a direct 256-bit (32-byte) occupancy mask for every group

    hybrid
        one selector bit per group, then either the tree or the 256-bit bitmap,
        whichever is smaller for that group

All layouts use direct per-group indexes containing the exact occupancy-stream
bit offset and retained-code ordinal. Runtime lookup never scans earlier trees
or computes a prefix sum.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CodecSpec:
    name: str
    code_bits: int
    native_values: int
    native_bytes: int
    aux_per_native_block: int


# Exact sparse-code metadata. Dense tensors do not need an entry here: every
# model tensor is still represented in the global type table.
CODECS: dict[str, CodecSpec] = {
    "F32":  CodecSpec("F32", 32, 1, 4, 0),
    "F16":  CodecSpec("F16", 16, 1, 2, 0),
    "BF16": CodecSpec("BF16", 16, 1, 2, 0),
    "Q4_0": CodecSpec("Q4_0", 4, 32, 18, 2),
    "Q4_1": CodecSpec("Q4_1", 4, 32, 20, 4),
    "Q5_0": CodecSpec("Q5_0", 5, 32, 22, 2),
    "Q5_1": CodecSpec("Q5_1", 5, 32, 24, 4),
    "Q8_0": CodecSpec("Q8_0", 8, 32, 34, 2),
    "Q8_1": CodecSpec("Q8_1", 8, 32, 36, 4),
    "Q2_K": CodecSpec("Q2_K", 2, 256, 84, 20),
    "Q3_K": CodecSpec("Q3_K", 3, 256, 110, 14),
    "Q4_K": CodecSpec("Q4_K", 4, 256, 144, 16),
    "Q5_K": CodecSpec("Q5_K", 5, 256, 176, 16),
    "Q6_K": CodecSpec("Q6_K", 6, 256, 210, 18),
    "Q8_K": CodecSpec("Q8_K", 8, 256, 292, 36),
}


@dataclass(frozen=True)
class TensorInfo:
    index: int
    name: str
    raw_type_id: int
    dtype: str
    dims: tuple[int, ...]
    encoded_bytes: int
    expert_bytes: int
    native_values: int
    native_bytes: int
    is_expert: bool

    @property
    def experts(self) -> int:
        return self.dims[2] if self.is_expert and len(self.dims) >= 3 else 0

    @property
    def weights_per_expert(self) -> int:
        return self.dims[0] * self.dims[1] if self.is_expert and len(self.dims) >= 2 else 0

    @property
    def dense_bpw(self) -> float:
        return 8.0 * self.expert_bytes / self.weights_per_expert


@dataclass
class ExpertStats:
    tensor_index: int
    expert: int
    weights: int
    retained: int
    groups: int
    empty_groups: int

    tree_bits: int
    bitmap_bits: int
    hybrid_location_bits: int
    hybrid_selector_bits: int
    hybrid_tree_groups: int
    hybrid_bitmap_groups: int

    tree_location_bytes: int
    bitmap_location_bytes: int
    hybrid_location_bytes: int
    hybrid_selector_bytes: int

    retained_pack_bytes: int
    retained_exact_bytes: int
    aux_bytes: int

    tree_padding_bytes: int
    bitmap_padding_bytes: int
    hybrid_padding_bytes: int


@dataclass
class TensorTotals:
    tensor: TensorInfo
    completed_experts: int = 0
    weights: int = 0
    retained: int = 0
    groups: int = 0
    empty_groups: int = 0

    tree_bits: int = 0
    bitmap_bits: int = 0
    hybrid_location_bits: int = 0
    hybrid_selector_bits: int = 0
    hybrid_tree_groups: int = 0
    hybrid_bitmap_groups: int = 0

    tree_location_bytes: int = 0
    bitmap_location_bytes: int = 0
    hybrid_location_bytes: int = 0
    hybrid_selector_bytes: int = 0

    retained_pack_bytes: int = 0
    retained_exact_bytes: int = 0
    aux_bytes: int = 0

    tree_padding_bytes: int = 0
    bitmap_padding_bytes: int = 0
    hybrid_padding_bytes: int = 0

    groups_per_expert: int = 0
    max_tree_bits_per_expert: int = 0
    max_bitmap_bits_per_expert: int = 0
    max_hybrid_location_bits_per_expert: int = 0
    max_retained_per_expert: int = 0

    def add(self, s: ExpertStats) -> None:
        self.completed_experts += 1
        self.weights += s.weights
        self.retained += s.retained
        self.groups += s.groups
        self.empty_groups += s.empty_groups

        self.tree_bits += s.tree_bits
        self.bitmap_bits += s.bitmap_bits
        self.hybrid_location_bits += s.hybrid_location_bits
        self.hybrid_selector_bits += s.hybrid_selector_bits
        self.hybrid_tree_groups += s.hybrid_tree_groups
        self.hybrid_bitmap_groups += s.hybrid_bitmap_groups

        self.tree_location_bytes += s.tree_location_bytes
        self.bitmap_location_bytes += s.bitmap_location_bytes
        self.hybrid_location_bytes += s.hybrid_location_bytes
        self.hybrid_selector_bytes += s.hybrid_selector_bytes

        self.retained_pack_bytes += s.retained_pack_bytes
        self.retained_exact_bytes += s.retained_exact_bytes
        self.aux_bytes += s.aux_bytes

        self.tree_padding_bytes += s.tree_padding_bytes
        self.bitmap_padding_bytes += s.bitmap_padding_bytes
        self.hybrid_padding_bytes += s.hybrid_padding_bytes

        if self.groups_per_expert == 0:
            self.groups_per_expert = s.groups
        elif self.groups_per_expert != s.groups:
            raise RuntimeError(f"{self.tensor.name}: groups/expert changed")

        self.max_tree_bits_per_expert = max(
            self.max_tree_bits_per_expert, s.tree_bits)
        self.max_bitmap_bits_per_expert = max(
            self.max_bitmap_bits_per_expert, s.bitmap_bits)
        self.max_hybrid_location_bits_per_expert = max(
            self.max_hybrid_location_bits_per_expert, s.hybrid_location_bits)
        self.max_retained_per_expert = max(
            self.max_retained_per_expert, s.retained)


class Bridge:
    def __init__(self, model: Path, library: Path):
        self.lib = ctypes.CDLL(str(library))
        self._bind()
        self.handle = self.lib.sgsim_open(os.fsencode(model))
        if not self.handle:
            raise RuntimeError(f"failed to open model path: {model}")
        if self.error:
            error = self.error
            self.close()
            raise RuntimeError(error)

    def _bind(self) -> None:
        l = self.lib
        l.sgsim_open.argtypes = [ctypes.c_char_p]
        l.sgsim_open.restype = ctypes.c_void_p
        l.sgsim_close.argtypes = [ctypes.c_void_p]
        l.sgsim_error.argtypes = [ctypes.c_void_p]
        l.sgsim_error.restype = ctypes.c_char_p
        l.sgsim_file_size.argtypes = [ctypes.c_void_p]
        l.sgsim_file_size.restype = ctypes.c_uint64
        l.sgsim_tensor_count.argtypes = [ctypes.c_void_p]
        l.sgsim_tensor_count.restype = ctypes.c_uint64
        l.sgsim_tensor_name.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        l.sgsim_tensor_name.restype = ctypes.c_char_p
        l.sgsim_tensor_is_expert.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        l.sgsim_tensor_is_expert.restype = ctypes.c_int
        l.sgsim_tensor_type.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        l.sgsim_tensor_type.restype = ctypes.c_uint32
        l.sgsim_tensor_type_name.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        l.sgsim_tensor_type_name.restype = ctypes.c_char_p
        l.sgsim_tensor_ndim.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        l.sgsim_tensor_ndim.restype = ctypes.c_uint32
        l.sgsim_tensor_dim.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
        l.sgsim_tensor_dim.restype = ctypes.c_uint64
        l.sgsim_tensor_block_values.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        l.sgsim_tensor_block_values.restype = ctypes.c_uint32
        l.sgsim_tensor_block_bytes.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        l.sgsim_tensor_block_bytes.restype = ctypes.c_uint32
        l.sgsim_tensor_encoded_bytes.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        l.sgsim_tensor_encoded_bytes.restype = ctypes.c_uint64
        l.sgsim_tensor_expert_bytes.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        l.sgsim_tensor_expert_bytes.restype = ctypes.c_uint64
        l.sgsim_load_expert.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
        ]
        l.sgsim_load_expert.restype = ctypes.c_int

    @property
    def error(self) -> str:
        raw = self.lib.sgsim_error(self.handle)
        return raw.decode("utf-8", errors="replace") if raw else ""

    @property
    def file_size(self) -> int:
        return int(self.lib.sgsim_file_size(self.handle))

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.sgsim_close(self.handle)
            self.handle = None

    def __enter__(self) -> "Bridge":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def all_tensors(self) -> list[TensorInfo]:
        out: list[TensorInfo] = []
        count = int(self.lib.sgsim_tensor_count(self.handle))
        for i in range(count):
            raw_name = self.lib.sgsim_tensor_name(self.handle, i)
            raw_type = self.lib.sgsim_tensor_type_name(self.handle, i)
            if not raw_name or not raw_type:
                raise RuntimeError(f"tensor {i}: missing name or type")
            ndim = int(self.lib.sgsim_tensor_ndim(self.handle, i))
            is_expert = bool(self.lib.sgsim_tensor_is_expert(self.handle, i))
            out.append(TensorInfo(
                index=i,
                name=raw_name.decode("utf-8", errors="replace"),
                raw_type_id=int(self.lib.sgsim_tensor_type(self.handle, i)),
                dtype=raw_type.decode("ascii", errors="replace"),
                dims=tuple(int(self.lib.sgsim_tensor_dim(self.handle, i, d)) for d in range(ndim)),
                encoded_bytes=int(self.lib.sgsim_tensor_encoded_bytes(self.handle, i)),
                expert_bytes=(int(self.lib.sgsim_tensor_expert_bytes(self.handle, i)) if is_expert else 0),
                native_values=int(self.lib.sgsim_tensor_block_values(self.handle, i)),
                native_bytes=int(self.lib.sgsim_tensor_block_bytes(self.handle, i)),
                is_expert=is_expert,
            ))
        return out

    def load_expert(self, tensor_index: int, expert: int, count: int) -> np.ndarray:
        out = np.empty(count, dtype=np.float32)
        ok = self.lib.sgsim_load_expert(
            self.handle, tensor_index, expert,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count,
        )
        if not ok:
            raise RuntimeError(self.error or "expert dequantization failed")
        return out


_WORKER: Bridge | None = None
_WORKER_TENSORS: dict[int, TensorInfo] = {}
_WORKER_THRESHOLD = 0.01
_WORKER_GROUP = 256
_WORKER_ALIGN = 8


def init_worker(model: str, library: str, tensors: list[TensorInfo], threshold: float,
                group: int, align: int) -> None:
    global _WORKER, _WORKER_TENSORS, _WORKER_THRESHOLD, _WORKER_GROUP, _WORKER_ALIGN
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _WORKER = Bridge(Path(model), Path(library))
    _WORKER_TENSORS = {t.index: t for t in tensors}
    _WORKER_THRESHOLD = threshold
    _WORKER_GROUP = group
    _WORKER_ALIGN = align


def tree_cost_power2_rows(bits_2d: np.ndarray) -> np.ndarray:
    values = np.asarray(bits_2d, dtype=np.bool_)
    width = values.shape[1]
    uniform = np.ones(values.shape, dtype=np.bool_)
    cost = np.full(values.shape, 2, dtype=np.int64)
    while width > 1:
        left_v, right_v = values[:, 0::2], values[:, 1::2]
        left_u, right_u = uniform[:, 0::2], uniform[:, 1::2]
        parent_u = left_u & right_u & (left_v == right_v)
        parent_c = np.where(parent_u, 2, 1 + cost[:, 0::2] + cost[:, 1::2])
        values = left_v
        uniform = parent_u
        cost = parent_c
        width //= 2
    return cost[:, 0]


def block_stats(mask: np.ndarray, group: int) -> tuple[np.ndarray, np.ndarray]:
    if mask.size % group:
        raise RuntimeError("expert weight count is not divisible by tree group")
    rows = np.asarray(mask, dtype=np.bool_).reshape(-1, group)
    if group & (group - 1):
        raise RuntimeError("tree group must be a power of two")
    return tree_cost_power2_rows(rows), np.count_nonzero(rows, axis=1).astype(np.int64)


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def bits_needed(max_inclusive: int) -> int:
    return 0 if max_inclusive <= 0 else int(max_inclusive).bit_length()


def packed8_bytes(count: int, code_bits: int) -> int:
    # Eight retained logical codes always occupy code_bits bytes.
    return ((count + 7) // 8) * code_bits


def exact_packed_bytes(count: int, code_bits: int) -> int:
    return (count * code_bits + 7) // 8


def analyze_job(job: tuple[int, int]) -> ExpertStats:
    if _WORKER is None:
        raise RuntimeError("worker not initialized")

    tensor_index, expert = job
    t = _WORKER_TENSORS[tensor_index]
    codec = CODECS[t.dtype]

    weights = _WORKER.load_expert(
        tensor_index, expert, t.weights_per_expert)

    mask = np.abs(weights) >= _WORKER_THRESHOLD
    costs, nnz = block_stats(mask, _WORKER_GROUP)

    groups = int(costs.size)
    retained = int(np.sum(nnz))

    tree_bits = int(np.sum(costs))
    bitmap_bits = groups * _WORKER_GROUP

    choose_tree = costs < _WORKER_GROUP
    hybrid_tree_groups = int(np.count_nonzero(choose_tree))
    hybrid_bitmap_groups = groups - hybrid_tree_groups
    hybrid_location_bits = int(np.sum(
        np.where(choose_tree, costs, _WORKER_GROUP)))
    hybrid_selector_bits = groups

    tree_location_bytes = (tree_bits + 7) // 8
    bitmap_location_bytes = bitmap_bits // 8
    hybrid_location_bytes = (hybrid_location_bits + 7) // 8
    hybrid_selector_bytes = (hybrid_selector_bits + 7) // 8

    native_per_group = _WORKER_GROUP // codec.native_values
    aux_bytes = (
        groups * native_per_group * codec.aux_per_native_block)

    pack_bytes = packed8_bytes(retained, codec.code_bits)
    exact_bytes = exact_packed_bytes(retained, codec.code_bits)

    tree_raw = tree_location_bytes + aux_bytes + pack_bytes
    tree_padded = (
        align_up(tree_location_bytes, _WORKER_ALIGN)
        + align_up(aux_bytes, _WORKER_ALIGN)
        + align_up(pack_bytes, _WORKER_ALIGN)
    )

    bitmap_raw = bitmap_location_bytes + aux_bytes + pack_bytes
    bitmap_padded = (
        align_up(bitmap_location_bytes, _WORKER_ALIGN)
        + align_up(aux_bytes, _WORKER_ALIGN)
        + align_up(pack_bytes, _WORKER_ALIGN)
    )

    hybrid_raw = (
        hybrid_selector_bytes
        + hybrid_location_bytes
        + aux_bytes
        + pack_bytes
    )
    hybrid_padded = (
        align_up(hybrid_selector_bytes, _WORKER_ALIGN)
        + align_up(hybrid_location_bytes, _WORKER_ALIGN)
        + align_up(aux_bytes, _WORKER_ALIGN)
        + align_up(pack_bytes, _WORKER_ALIGN)
    )

    return ExpertStats(
        tensor_index=tensor_index,
        expert=expert,
        weights=int(mask.size),
        retained=retained,
        groups=groups,
        empty_groups=int(np.count_nonzero(nnz == 0)),

        tree_bits=tree_bits,
        bitmap_bits=bitmap_bits,
        hybrid_location_bits=hybrid_location_bits,
        hybrid_selector_bits=hybrid_selector_bits,
        hybrid_tree_groups=hybrid_tree_groups,
        hybrid_bitmap_groups=hybrid_bitmap_groups,

        tree_location_bytes=tree_location_bytes,
        bitmap_location_bytes=bitmap_location_bytes,
        hybrid_location_bytes=hybrid_location_bytes,
        hybrid_selector_bytes=hybrid_selector_bytes,

        retained_pack_bytes=pack_bytes,
        retained_exact_bytes=exact_bytes,
        aux_bytes=aux_bytes,

        tree_padding_bytes=tree_padded - tree_raw,
        bitmap_padding_bytes=bitmap_padded - bitmap_raw,
        hybrid_padding_bytes=hybrid_padded - hybrid_raw,
    )


def human_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    x = float(value)
    for unit in units:
        if abs(x) < 1024.0 or unit == units[-1]:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TiB"


def parse_name(name: str) -> tuple[int, str] | None:
    m = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", name)
    return (int(m.group(1)), m.group(2)) if m else None


def build_global_type_table(all_tensors: list[TensorInfo]) -> dict[str, Any]:
    by_raw: dict[int, list[TensorInfo]] = defaultdict(list)

    for t in all_tensors:
        by_raw[t.raw_type_id].append(t)

    ordered_raw = sorted(by_raw)
    compact = {raw: i for i, raw in enumerate(ordered_raw)}
    count = len(ordered_raw)

    compact_id_bits = 0 if count <= 1 else math.ceil(math.log2(count))
    max_raw_type = max(ordered_raw, default=0)
    raw_type_bits = bits_needed(max_raw_type)

    rows = []
    for raw in ordered_raw:
        ts = by_raw[raw]
        names = sorted({t.dtype for t in ts})

        if len(names) != 1:
            raise RuntimeError(
                f"raw type {raw} has inconsistent names: {names}")

        first = ts[0]
        codec = CODECS.get(first.dtype)

        rows.append({
            "compact_id": compact[raw],
            "raw_ggml_type": raw,
            "name": first.dtype,
            "tensor_count": len(ts),
            "encoded_bytes": sum(t.encoded_bytes for t in ts),
            "native_values": first.native_values,
            "native_bytes": first.native_bytes,
            "sparse_code_bits": (
                codec.code_bits if codec else None),
            "aux_per_native_block": (
                codec.aux_per_native_block if codec else None),
        })

    # Compact header:
    #   uint8 type_count
    #   uint8 raw_type_bits
    #   uint8 compact_id_bits
    #   uint8 reserved
    #   bit-packed raw GGML type IDs
    #
    # Names, block sizes and codec traits are not repeated in the file.
    # The reader obtains them from the raw GGML type registry.
    table_header_bytes = 4
    table_payload_bits = count * raw_type_bits
    table_payload_bytes = (table_payload_bits + 7) // 8
    table_bytes = table_header_bytes + table_payload_bytes

    tensor_type_id_bits = len(all_tensors) * compact_id_bits
    tensor_type_id_stream_bytes = (
        tensor_type_id_bits + 7) // 8

    return {
        "count": count,
        "compact_id_bits_per_tensor": compact_id_bits,
        "raw_type_bits_in_table": raw_type_bits,
        "table_header_bytes": table_header_bytes,
        "table_payload_bits": table_payload_bits,
        "table_payload_bytes": table_payload_bytes,
        "table_bytes": table_bytes,
        "tensor_type_id_bits": tensor_type_id_bits,
        "tensor_type_id_stream_bytes": tensor_type_id_stream_bytes,
        "rows": rows,
    }


def validate_sparse_tensors(tensors: list[TensorInfo], group: int) -> dict[str, Any]:
    if not tensors:
        raise RuntimeError("no routed MoE expert tensors found")
    errors: list[str] = []
    by_layer: dict[int, dict[str, TensorInfo]] = defaultdict(dict)
    for t in tensors:
        parsed = parse_name(t.name)
        if parsed is None:
            errors.append(f"unrecognized routed expert tensor name: {t.name}")
            continue
        layer, projection = parsed
        by_layer[layer][projection] = t
        codec = CODECS.get(t.dtype)
        if codec is None:
            errors.append(f"{t.name}: no exact sparse codec metadata for {t.dtype}")
            continue
        if (t.native_values, t.native_bytes) != (codec.native_values, codec.native_bytes):
            errors.append(
                f"{t.name}: bridge says {t.native_values} values/{t.native_bytes} bytes, "
                f"codec says {codec.native_values}/{codec.native_bytes}")
        if group % codec.native_values:
            errors.append(f"{t.name}: 256-value tree group is incompatible with {t.dtype}")
        if t.weights_per_expert % group:
            errors.append(f"{t.name}: weights/expert is not divisible by {group}")

    for layer, projections in sorted(by_layer.items()):
        missing = {"gate", "up", "down"} - set(projections)
        if missing:
            errors.append(f"layer {layer} missing {sorted(missing)}")
        counts = {t.experts for t in projections.values()}
        if len(counts) > 1:
            errors.append(f"layer {layer} projection expert counts differ: {sorted(counts)}")

    if errors:
        raise RuntimeError("validation failed before scan:\n  - " + "\n  - ".join(errors))

    hist = Counter(t.dtype for t in tensors)
    return {
        "layers": sorted(by_layer),
        "type_histogram": dict(sorted(hist.items())),
    }


def tensor_index_layout(
    tt: TensorTotals,
    mode: str,
    completed_only: bool = False,
) -> dict[str, int]:
    experts = (
        tt.completed_experts
        if completed_only
        else tt.tensor.experts
    )

    if experts == 0 or tt.groups_per_expert == 0:
        return {
            "location_offset_bits": 0,
            "value_ordinal_bits": 0,
            "entry_bits": 0,
            "entries": 0,
            "bytes": 0,
        }

    if mode == "tree":
        max_location_bits = tt.max_tree_bits_per_expert
    elif mode == "bitmap":
        max_location_bits = tt.max_bitmap_bits_per_expert
    elif mode == "hybrid":
        max_location_bits = tt.max_hybrid_location_bits_per_expert
    else:
        raise ValueError(f"unknown occupancy mode: {mode}")

    location_bits = bits_needed(max_location_bits)
    value_bits = bits_needed(tt.max_retained_per_expert)
    entry_bits = location_bits + value_bits

    entries = experts * (tt.groups_per_expert + 1)

    return {
        "location_offset_bits": location_bits,
        "value_ordinal_bits": value_bits,
        "entry_bits": entry_bits,
        "entries": entries,
        "bytes": align_up(
            (entries * entry_bits + 7) // 8,
            8,
        ),
    }


def sum_attr(items: list[TensorTotals], name: str) -> int:
    return sum(int(getattr(x, name)) for x in items)


def stream_bytes(
    items: list[TensorTotals],
    mode: str,
    use_pack8: bool = True,
) -> int:
    values = sum_attr(
        items,
        "retained_pack_bytes"
        if use_pack8
        else "retained_exact_bytes",
    )

    common = sum_attr(items, "aux_bytes") + values

    if mode == "tree":
        return (
            common
            + sum_attr(items, "tree_location_bytes")
            + sum_attr(items, "tree_padding_bytes")
        )

    if mode == "bitmap":
        return (
            common
            + sum_attr(items, "bitmap_location_bytes")
            + sum_attr(items, "bitmap_padding_bytes")
        )

    if mode == "hybrid":
        return (
            common
            + sum_attr(items, "hybrid_selector_bytes")
            + sum_attr(items, "hybrid_location_bytes")
            + sum_attr(items, "hybrid_padding_bytes")
        )

    raise ValueError(f"unknown occupancy mode: {mode}")


def direct_index_bytes(
    items: list[TensorTotals],
    mode: str,
    completed_only: bool = False,
) -> int:
    return sum(
        tensor_index_layout(
            tt,
            mode,
            completed_only,
        )["bytes"]
        for tt in items
    )


def expert_directory_bytes(
    items: list[TensorTotals],
    completed_only: bool = False,
) -> int:
    # Three direct absolute 64-bit stream bases:
    # occupancy, auxiliary and retained values.
    count = sum(
        tt.completed_experts
        if completed_only
        else tt.tensor.experts
        for tt in items
    )
    return count * 24


def sparse_tensor_descriptor_bytes(
    items: list[TensorTotals],
    completed_only: bool = False,
) -> int:
    count = sum(
        1
        for tt in items
        if (
            tt.completed_experts
            if completed_only
            else tt.tensor.experts
        ) > 0
    )
    return count * 64


def compact_bytes(
    items: list[TensorTotals],
    mode: str,
    completed_only: bool = False,
    use_pack8: bool = True,
) -> int:
    return (
        stream_bytes(items, mode, use_pack8)
        + direct_index_bytes(
            items,
            mode,
            completed_only,
        )
        + expert_directory_bytes(
            items,
            completed_only,
        )
        + sparse_tensor_descriptor_bytes(
            items,
            completed_only,
        )
    )


def compression_text(before: int, after: int) -> str:
    ratio = after / before if before else 0.0
    saved = before - after
    return (
        f"{human_bytes(before)} -> {human_bytes(after)} "
        f"ratio={ratio:.5f} saved={100.0*(1.0-ratio):.3f}% ({human_bytes(saved)})"
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model", type=Path)
    p.add_argument(
        "--library",
        type=Path,
        default=Path(__file__).with_name(
            "libsgguf_compact_sim.so"),
    )
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--stream-align", type=int, default=8)
    p.add_argument("--jobs", default="auto")
    p.add_argument("--trunk-only", action="store_true")
    p.add_argument("--report-every", type=int, default=128)
    p.add_argument("--csv", type=Path)
    p.add_argument("--json", type=Path)
    args = p.parse_args()

    group = 256

    if not args.model.is_file():
        p.error(f"model not found: {args.model}")

    if not args.library.is_file():
        p.error(
            f"library not found: {args.library}; "
            "run ./build_sgguf_compact_sim.sh"
        )

    if (
        args.stream_align < 1
        or args.stream_align
        & (args.stream_align - 1)
    ):
        p.error("--stream-align must be a power of two")

    if (
        not math.isfinite(args.threshold)
        or args.threshold < 0
    ):
        p.error(
            "--threshold must be finite and non-negative")

    if args.report_every < 1:
        p.error("--report-every must be positive")

    jobs_count = (
        min(os.cpu_count() or 1, 16)
        if args.jobs == "auto"
        else int(args.jobs)
    )

    if jobs_count < 1:
        p.error("--jobs must be positive")

    with Bridge(args.model, args.library) as bridge:
        file_size = bridge.file_size
        all_tensors = bridge.all_tensors()

    type_table = build_global_type_table(all_tensors)

    expert_tensors = [
        t
        for t in all_tensors
        if t.is_expert and len(t.dims) == 3
    ]

    if args.trunk_only and expert_tensors:
        parsed_layers = [
            parse_name(t.name)[0]
            for t in expert_tensors
            if parse_name(t.name)
        ]

        if parsed_layers:
            highest = max(parsed_layers)
            expert_tensors = [
                t
                for t in expert_tensors
                if not t.name.startswith(
                    f"blk.{highest}.")
            ]

    validation = validate_sparse_tensors(
        expert_tensors,
        group,
    )

    layers = validation["layers"]
    sparse_source_bytes = sum(
        t.encoded_bytes
        for t in expert_tensors
    )
    base_non_sparse_bytes = (
        file_size - sparse_source_bytes
    )
    total_jobs = sum(
        t.experts
        for t in expert_tensors
    )

    print(f"model={args.model}")
    print(
        f"source_file={human_bytes(file_size)} "
        f"tensors={len(all_tensors)}"
    )
    print(
        "global_types="
        f"{type_table['count']} "
        "compact_type_bits_per_tensor="
        f"{type_table['compact_id_bits_per_tensor']} "
        "raw_type_bits_in_header_table="
        f"{type_table['raw_type_bits_in_table']}"
    )
    print("GLOBAL TYPE TABLE")

    for row in type_table["rows"]:
        sparse = (
            f" code={row['sparse_code_bits']}b "
            f"aux/native="
            f"{row['aux_per_native_block']}B"
            if row["sparse_code_bits"] is not None
            else " dense-only/no exact sparse codec"
        )

        print(
            f"  id={row['compact_id']:>2} "
            f"raw={row['raw_ggml_type']:>3} "
            f"{row['name']:<10} "
            f"tensors={row['tensor_count']:>4} "
            f"bytes="
            f"{human_bytes(row['encoded_bytes']):>10} "
            f"native="
            f"{row['native_values']}w/"
            f"{row['native_bytes']}B"
            f"{sparse}"
        )

    print(
        "global_type_table="
        f"{human_bytes(type_table['table_bytes'])} "
        f"(4-byte header + "
        f"{type_table['table_payload_bits']} packed bits)"
    )
    print(
        "all_tensor_type_ids="
        f"{human_bytes(type_table['tensor_type_id_stream_bytes'])} "
        f"({type_table['tensor_type_id_bits']} packed bits)"
    )

    print(
        f"routed_sparse_tensors="
        f"{len(expert_tensors)} "
        f"layers={len(layers)} "
        f"range={layers[0]}..{layers[-1]}"
    )
    print(
        "expert_type_histogram="
        f"{validation['type_histogram']}"
    )
    print(
        f"expert_jobs={total_jobs:,} "
        f"threshold={args.threshold:g} "
        "occupancy_group=256"
    )
    print(
        "retained_pack=8 codes; "
        "physical pack bytes equal codec bit width"
    )
    print(
        f"workers={jobs_count} "
        "no page-block count; "
        "direct index stores occupancy-bit offset "
        "+ retained-code ordinal"
    )
    print(
        "occupancy candidates: "
        "tree, fixed 256-bit bitmap, "
        "1-bit selector hybrid"
    )

    totals = {
        t.index: TensorTotals(t)
        for t in expert_tensors
    }
    jobs = [
        (t.index, e)
        for t in expert_tensors
        for e in range(t.experts)
    ]

    start = time.monotonic()
    done = 0

    csv_file = None
    writer = None

    if args.csv:
        csv_file = args.csv.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "tensor",
                "dtype",
                "expert",
                "weights",
                "retained",
                "retained_pct",
                "groups",
                "tree_bits",
                "bitmap_bits",
                "hybrid_location_bits",
                "hybrid_selector_bits",
                "hybrid_tree_groups",
                "hybrid_bitmap_groups",
                "tree_location_bytes",
                "bitmap_location_bytes",
                "hybrid_location_bytes",
                "hybrid_selector_bytes",
                "retained_pack_bytes",
                "retained_exact_bytes",
                "aux_bytes",
                "tree_padding_bytes",
                "bitmap_padding_bytes",
                "hybrid_padding_bytes",
            ],
        )
        writer.writeheader()

    ctx = mp.get_context("fork")

    try:
        with ctx.Pool(
            jobs_count,
            initializer=init_worker,
            initargs=(
                str(args.model),
                str(args.library),
                expert_tensors,
                args.threshold,
                group,
                args.stream_align,
            ),
        ) as pool:
            for s in pool.imap_unordered(
                analyze_job,
                jobs,
                chunksize=1,
            ):
                totals[s.tensor_index].add(s)
                done += 1

                if writer:
                    t = totals[s.tensor_index].tensor
                    writer.writerow({
                        "tensor": t.name,
                        "dtype": t.dtype,
                        "expert": s.expert,
                        "weights": s.weights,
                        "retained": s.retained,
                        "retained_pct": (
                            100.0
                            * s.retained
                            / s.weights
                        ),
                        "groups": s.groups,
                        "tree_bits": s.tree_bits,
                        "bitmap_bits": s.bitmap_bits,
                        "hybrid_location_bits":
                            s.hybrid_location_bits,
                        "hybrid_selector_bits":
                            s.hybrid_selector_bits,
                        "hybrid_tree_groups":
                            s.hybrid_tree_groups,
                        "hybrid_bitmap_groups":
                            s.hybrid_bitmap_groups,
                        "tree_location_bytes":
                            s.tree_location_bytes,
                        "bitmap_location_bytes":
                            s.bitmap_location_bytes,
                        "hybrid_location_bytes":
                            s.hybrid_location_bytes,
                        "hybrid_selector_bytes":
                            s.hybrid_selector_bytes,
                        "retained_pack_bytes":
                            s.retained_pack_bytes,
                        "retained_exact_bytes":
                            s.retained_exact_bytes,
                        "aux_bytes": s.aux_bytes,
                        "tree_padding_bytes":
                            s.tree_padding_bytes,
                        "bitmap_padding_bytes":
                            s.bitmap_padding_bytes,
                        "hybrid_padding_bytes":
                            s.hybrid_padding_bytes,
                    })
                    csv_file.flush()

                if (
                    done % args.report_every == 0
                    or done == total_jobs
                ):
                    items = list(totals.values())

                    elapsed = (
                        time.monotonic() - start
                    )
                    rate = (
                        done / elapsed
                        if elapsed
                        else 0.0
                    )
                    eta = (
                        (total_jobs - done) / rate
                        if rate
                        else 0.0
                    )

                    weights = sum_attr(
                        items,
                        "weights",
                    )
                    retained = sum_attr(
                        items,
                        "retained",
                    )

                    processed_source = sum(
                        tt.completed_experts
                        * tt.tensor.expert_bytes
                        for tt in items
                    )

                    sizes = {
                        mode: compact_bytes(
                            items,
                            mode,
                            completed_only=True,
                            use_pack8=True,
                        )
                        for mode in (
                            "tree",
                            "bitmap",
                            "hybrid",
                        )
                    }

                    best_mode = min(
                        sizes,
                        key=sizes.get,
                    )
                    best_size = sizes[best_mode]

                    ratio = (
                        best_size / processed_source
                        if processed_source
                        else 0.0
                    )

                    remaining_source = max(
                        0,
                        sparse_source_bytes
                        - processed_source,
                    )
                    projected_sparse = (
                        best_size
                        + int(round(
                            remaining_source
                            * ratio
                        ))
                    )

                    global_header = (
                        type_table["table_bytes"]
                        + type_table[
                            "tensor_type_id_stream_bytes"
                        ]
                    )

                    projected_model = (
                        base_non_sparse_bytes
                        + projected_sparse
                        + global_header
                    )

                    retained_pct = (
                        100.0 * retained / weights
                        if weights
                        else 0.0
                    )

                    print(
                        f"[SIM {done:>6}/{total_jobs} "
                        f"{100.0*done/total_jobs:6.2f}%] "
                        f"rate={rate:.2f}/s "
                        f"eta={eta/60:.1f}m "
                        f"retained={retained_pct:.4f}% "
                        f"source="
                        f"{human_bytes(processed_source)} "
                        f"tree="
                        f"{human_bytes(sizes['tree'])} "
                        f"bitmap="
                        f"{human_bytes(sizes['bitmap'])} "
                        f"hybrid="
                        f"{human_bytes(sizes['hybrid'])} "
                        f"best={best_mode}:"
                        f"{human_bytes(best_size)} "
                        f"ratio={ratio:.5f} "
                        f"projected_model="
                        f"{human_bytes(projected_model)}",
                        flush=True,
                    )
    finally:
        if csv_file:
            csv_file.close()

    items = list(totals.values())

    weights = sum_attr(items, "weights")
    retained = sum_attr(items, "retained")
    groups = sum_attr(items, "groups")

    aux_bytes = sum_attr(items, "aux_bytes")
    value_pack_bytes = sum_attr(
        items,
        "retained_pack_bytes",
    )
    value_exact_bytes = sum_attr(
        items,
        "retained_exact_bytes",
    )

    global_header_overhead = (
        type_table["table_bytes"]
        + type_table[
            "tensor_type_id_stream_bytes"
        ]
    )

    modes: dict[str, dict[str, int | float]] = {}

    for mode in ("tree", "bitmap", "hybrid"):
        compact_sparse = compact_bytes(
            items,
            mode,
            use_pack8=True,
        )
        compact_sparse_exact = compact_bytes(
            items,
            mode,
            use_pack8=False,
        )
        whole_model = (
            base_non_sparse_bytes
            + compact_sparse
            + global_header_overhead
        )

        modes[mode] = {
            "compact_sparse": compact_sparse,
            "compact_sparse_exact":
                compact_sparse_exact,
            "whole_model": whole_model,
            "ratio": (
                compact_sparse
                / sparse_source_bytes
            ),
        }

    best_mode = min(
        modes,
        key=lambda mode: modes[mode][
            "compact_sparse"
        ],
    )

    tree_chosen = sum_attr(
        items,
        "hybrid_tree_groups",
    )
    bitmap_chosen = sum_attr(
        items,
        "hybrid_bitmap_groups",
    )

    old_theoretical_bits = sum(
        tt.tree_bits
        + tt.retained
        * tt.tensor.dense_bpw
        for tt in items
    )
    old_theoretical_bytes = math.ceil(
        old_theoretical_bits / 8
    )

    print("\n" + "=" * 104)
    print("FINAL COMPACT LAYOUT COMPARISON")
    print(
        f"  source routed experts          "
        f"{human_bytes(sparse_source_bytes)}"
    )
    print(
        f"  source whole GGUF              "
        f"{human_bytes(file_size)}"
    )

    for mode in (
        "tree",
        "bitmap",
        "hybrid",
    ):
        compact_sparse = int(
            modes[mode]["compact_sparse"]
        )
        whole_model = int(
            modes[mode]["whole_model"]
        )

        print(
            f"\n  {mode.upper()}"
        )
        print(
            f"    routed experts               "
            f"{human_bytes(compact_sparse)} "
            f"ratio="
            f"{compact_sparse/sparse_source_bytes:.6f} "
            f"saved="
            f"{100.0*(1.0-compact_sparse/sparse_source_bytes):.3f}%"
        )
        print(
            f"    estimated whole SGGUF        "
            f"{human_bytes(whole_model)} "
            f"ratio={whole_model/file_size:.6f} "
            f"saved="
            f"{100.0*(1.0-whole_model/file_size):.3f}%"
        )

    print(
        f"\n  BEST MODE                     "
        f"{best_mode.upper()}"
    )

    print("\nOCCUPANCY LOCATION COMPONENTS")
    print(
        f"  tree stream                    "
        f"{human_bytes(sum_attr(items, 'tree_location_bytes'))}"
    )
    print(
        f"  fixed 256-bit bitmap stream    "
        f"{human_bytes(sum_attr(items, 'bitmap_location_bytes'))}"
    )
    print(
        f"  hybrid chosen-location stream  "
        f"{human_bytes(sum_attr(items, 'hybrid_location_bytes'))}"
    )
    print(
        f"  hybrid 1-bit selectors         "
        f"{human_bytes(sum_attr(items, 'hybrid_selector_bytes'))}"
    )
    print(
        f"  hybrid chose tree              "
        f"{tree_chosen:,}/{groups:,} groups"
    )
    print(
        f"  hybrid chose bitmap            "
        f"{bitmap_chosen:,}/{groups:,} groups"
    )

    hybrid_vs_tree = (
        int(modes["hybrid"]["compact_sparse"])
        - int(modes["tree"]["compact_sparse"])
    )

    print(
        "  hybrid minus always-tree       "
        f"{human_bytes(hybrid_vs_tree)} "
        f"({'hybrid larger' if hybrid_vs_tree >= 0 else 'hybrid smaller'})"
    )

    print("\nCOMMON COMPONENTS")
    print(
        f"  retained 8-code packs          "
        f"{human_bytes(value_pack_bytes)}"
    )
    print(
        f"  fully continuous retained bits "
        f"{human_bytes(value_exact_bytes)}"
    )
    print(
        f"  8-code pack overhead           "
        f"{human_bytes(value_pack_bytes-value_exact_bytes)}"
    )
    print(
        f"  type auxiliary streams         "
        f"{human_bytes(aux_bytes)}"
    )
    print(
        f"  expert direct-offset tables    "
        f"{human_bytes(expert_directory_bytes(items))}"
    )
    print(
        f"  sparse tensor descriptors      "
        f"{human_bytes(sparse_tensor_descriptor_bytes(items))}"
    )
    print(
        f"  global type table              "
        f"{human_bytes(type_table['table_bytes'])}"
    )
    print(
        f"  all tensor compact type IDs    "
        f"{human_bytes(type_table['tensor_type_id_stream_bytes'])}"
    )
    print(
        f"  old theoretical sparse region  "
        f"{human_bytes(old_theoretical_bytes)}"
    )

    print("\nDIRECT INDEX COST BY MODE")
    for mode in (
        "tree",
        "bitmap",
        "hybrid",
    ):
        print(
            f"  {mode:<7} "
            f"{human_bytes(direct_index_bytes(items, mode))}"
        )

    print("\nDIRECT INDEX WIDTHS BY SPARSE TENSOR")
    for tt in items:
        codec = CODECS[tt.tensor.dtype]

        tree_layout = tensor_index_layout(
            tt,
            "tree",
        )
        bitmap_layout = tensor_index_layout(
            tt,
            "bitmap",
        )
        hybrid_layout = tensor_index_layout(
            tt,
            "hybrid",
        )

        print(
            f"  {tt.tensor.name}: "
            f"type={tt.tensor.dtype} "
            f"code={codec.code_bits}b "
            f"pack=8x{codec.code_bits}b="
            f"{codec.code_bits}B "
            f"value_ordinal="
            f"{tree_layout['value_ordinal_bits']}b "
            f"location_off="
            f"tree:{tree_layout['location_offset_bits']}b/"
            f"bitmap:{bitmap_layout['location_offset_bits']}b/"
            f"hybrid:{hybrid_layout['location_offset_bits']}b "
            f"entry="
            f"tree:{tree_layout['entry_bits']}b/"
            f"bitmap:{bitmap_layout['entry_bits']}b/"
            f"hybrid:{hybrid_layout['entry_bits']}b"
        )

    print("\nDIRECT MMAP LOOKUP")
    print(
        "  compact_type_id -> "
        "global header type table -> raw GGML type"
    )
    print(
        "  expert entry -> exact occupancy, "
        "auxiliary and retained-value stream bases"
    )
    print(
        "  group index -> exact occupancy bit offset "
        "and retained-code ordinal"
    )
    print(
        "  hybrid selector bit -> tree or 256-bit bitmap"
    )
    print(
        "  no previous-tree scan, prefix sum, "
        "or fixed page-block count"
    )

    summary = {
        "model": str(args.model),
        "threshold": args.threshold,
        "occupancy_group": group,
        "global_type_table": type_table,
        "sparse_validation": validation,
        "source_file_bytes": file_size,
        "source_sparse_bytes": sparse_source_bytes,
        "base_non_sparse_bytes":
            base_non_sparse_bytes,
        "weights": weights,
        "retained": retained,
        "groups": groups,
        "hybrid_tree_groups": tree_chosen,
        "hybrid_bitmap_groups": bitmap_chosen,
        "best_mode": best_mode,
        "modes": modes,
        "common_components": {
            "retained_pack8_bytes":
                value_pack_bytes,
            "retained_exact_bytes":
                value_exact_bytes,
            "aux_bytes": aux_bytes,
            "expert_directory_bytes":
                expert_directory_bytes(items),
            "sparse_descriptor_bytes":
                sparse_tensor_descriptor_bytes(items),
            "global_type_table_bytes":
                type_table["table_bytes"],
            "tensor_type_id_stream_bytes":
                type_table[
                    "tensor_type_id_stream_bytes"
                ],
        },
        "occupancy_components": {
            "tree_location_bytes":
                sum_attr(
                    items,
                    "tree_location_bytes",
                ),
            "bitmap_location_bytes":
                sum_attr(
                    items,
                    "bitmap_location_bytes",
                ),
            "hybrid_location_bytes":
                sum_attr(
                    items,
                    "hybrid_location_bytes",
                ),
            "hybrid_selector_bytes":
                sum_attr(
                    items,
                    "hybrid_selector_bytes",
                ),
        },
        "tensor_index_layouts": {
            tt.tensor.name: {
                mode: tensor_index_layout(
                    tt,
                    mode,
                )
                for mode in (
                    "tree",
                    "bitmap",
                    "hybrid",
                )
            }
            for tt in items
        },
    }

    if args.json:
        args.json.write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"\nJSON written: {args.json}"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\ninterrupted",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
