#!/usr/bin/env python3
"""Interactive sorted-weight and exact-zero-mask tree viewer for GGUF MoE experts.

The script uses a tiny ctypes bridge linked against Colibri's own GGUF reader
and dequantizers. It loads exactly one expert at a time. In interactive mode,
closing the current Matplotlib window triggers loading of the next expert.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TensorInfo:
    index: int
    name: str
    dtype: str
    dims: tuple[int, int, int]

    @property
    def experts(self) -> int:
        return self.dims[2]

    @property
    def weights_per_expert(self) -> int:
        return self.dims[0] * self.dims[1]


class GgufWeights:
    def __init__(self, model: Path, library: Path):
        self._lib = ctypes.CDLL(str(library))
        self._bind()
        self._handle = self._lib.gwv_open(os.fsencode(model))
        if not self._handle:
            raise RuntimeError("could not allocate GGUF loader")
        error = self.error
        if error:
            self.close()
            raise RuntimeError(error)

    def _bind(self) -> None:
        lib = self._lib
        lib.gwv_open.argtypes = [ctypes.c_char_p]
        lib.gwv_open.restype = ctypes.c_void_p
        lib.gwv_close.argtypes = [ctypes.c_void_p]
        lib.gwv_close.restype = None
        lib.gwv_error.argtypes = [ctypes.c_void_p]
        lib.gwv_error.restype = ctypes.c_char_p
        lib.gwv_tensor_count.argtypes = [ctypes.c_void_p]
        lib.gwv_tensor_count.restype = ctypes.c_uint64
        lib.gwv_tensor_is_expert.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.gwv_tensor_is_expert.restype = ctypes.c_int
        lib.gwv_tensor_name.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.gwv_tensor_name.restype = ctypes.c_char_p
        lib.gwv_tensor_type_name.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.gwv_tensor_type_name.restype = ctypes.c_char_p
        lib.gwv_tensor_dim.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
        lib.gwv_tensor_dim.restype = ctypes.c_uint64
        lib.gwv_tensor_expert_count.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.gwv_tensor_expert_count.restype = ctypes.c_uint64
        lib.gwv_tensor_weights_per_expert.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.gwv_tensor_weights_per_expert.restype = ctypes.c_uint64
        lib.gwv_load_expert.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64,
        ]
        lib.gwv_load_expert.restype = ctypes.c_int

    @property
    def error(self) -> str:
        raw = self._lib.gwv_error(self._handle)
        return raw.decode("utf-8", errors="replace") if raw else ""

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.gwv_close(self._handle)
            self._handle = None

    def __enter__(self) -> "GgufWeights":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def expert_tensors(self) -> list[TensorInfo]:
        tensors: list[TensorInfo] = []
        count = int(self._lib.gwv_tensor_count(self._handle))
        for index in range(count):
            if not self._lib.gwv_tensor_is_expert(self._handle, index):
                continue
            name_raw = self._lib.gwv_tensor_name(self._handle, index)
            dtype_raw = self._lib.gwv_tensor_type_name(self._handle, index)
            name = name_raw.decode("utf-8", errors="replace")
            dtype = dtype_raw.decode("ascii", errors="replace")
            dims = tuple(int(self._lib.gwv_tensor_dim(self._handle, index, d)) for d in range(3))
            tensors.append(TensorInfo(index, name, dtype, dims))
        return tensors

    def load_expert(self, tensor: TensorInfo, expert: int) -> np.ndarray:
        weights = np.empty(tensor.weights_per_expert, dtype=np.float32)
        ok = self._lib.gwv_load_expert(
            self._handle,
            tensor.index,
            expert,
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            weights.size,
        )
        if not ok:
            raise RuntimeError(self.error or "unknown dequantization failure")
        return weights


def parse_expert_spec(spec: str, upper: int) -> list[int]:
    """Parse forms such as 0,2,7-10,20:30, or all."""
    if spec.strip().lower() == "all":
        return list(range(upper))
    values: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) not in (2, 3):
                raise ValueError(f"invalid range: {part}")
            start = int(pieces[0]) if pieces[0] else 0
            stop = int(pieces[1]) if pieces[1] else upper
            step = int(pieces[2]) if len(pieces) == 3 and pieces[2] else 1
            values.update(range(start, stop, step))
        elif "-" in part[1:]:
            start_s, stop_s = part.split("-", 1)
            start, stop = int(start_s), int(stop_s)
            values.update(range(start, stop + 1))
        else:
            values.add(int(part))
    result = sorted(values)
    bad = [x for x in result if x < 0 or x >= upper]
    if bad:
        raise ValueError(f"expert indices outside 0..{upper - 1}: {bad[:8]}")
    return result


def projection_pattern(projection: str | None) -> str | None:
    if not projection:
        return None
    return rf"\.ffn_{re.escape(projection)}_exps\.weight$"


def choose_tensors(
    tensors: list[TensorInfo], tensor_regex: str | None, layer: int | None, projection: str | None
) -> list[TensorInfo]:
    patterns: list[str] = []
    if tensor_regex:
        patterns.append(tensor_regex)
    if layer is not None:
        patterns.append(rf"^blk\.{layer}\.")
    pp = projection_pattern(projection)
    if pp:
        patterns.append(pp)
    chosen = [t for t in tensors if all(re.search(p, t.name) for p in patterns)]
    if not patterns:
        return tensors[:1]
    return chosen


def sparsity_text(weights: np.ndarray) -> str:
    a = np.abs(weights)
    return (
        f"zero={np.mean(weights == 0.0) * 100:.5f}%   "
        f"|w|≤1e-5={np.mean(a <= 1e-5) * 100:.5f}%   "
        f"≤1e-4={np.mean(a <= 1e-4) * 100:.5f}%   "
        f"≤1e-3={np.mean(a <= 1e-3) * 100:.5f}%   "
        f"≤1e-2={np.mean(a <= 1e-2) * 100:.5f}%"
    )


def tree_compression_cost(bits: np.ndarray) -> dict[str, int | float | bool]:
    """Cost of the user's binary uniform-range tree encoding.

    A uniform range is encoded as a two-bit leaf: one node marker plus the
    stored bit value. A mixed range costs one branch marker plus both children.
    This measures only the exact-zero/nonzero occupancy mask, not weight values.
    """
    bit_array = np.asarray(bits, dtype=np.uint8).reshape(-1)
    n = int(bit_array.size)
    if n == 0:
        return {
            "original_bits": 0,
            "tree_bits": 0,
            "ratio": 0.0,
            "effective": False,
        }

    # A Python integer prefix table makes the recursive range tests fast.
    prefix = [0] * (n + 1)
    total = 0
    for i, bit in enumerate(bit_array, start=1):
        total += int(bit)
        prefix[i] = total

    def visit(left: int, right: int) -> int:
        ones = prefix[right] - prefix[left]
        if ones == 0 or ones == right - left:
            return 2
        middle = (left + right) // 2
        return 1 + visit(left, middle) + visit(middle, right)

    tree_bits = visit(0, n)
    return {
        "original_bits": n,
        "tree_bits": tree_bits,
        "ratio": tree_bits / n,
        "effective": tree_bits < n,
    }


def tree_result_text(label: str, result: dict[str, int | float | bool]) -> str:
    effective = "YES" if bool(result["effective"]) else "NO"
    return (
        f"{label}: tree={int(result['tree_bits']):,} bits / "
        f"raw={int(result['original_bits']):,} bits, "
        f"ratio={float(result['ratio']):.6f}, effective={effective}"
    )


def decimate_sorted(values: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    n = values.size
    if n <= max_points:
        return np.arange(n, dtype=np.int64), values
    index = np.linspace(0, n - 1, max_points, dtype=np.int64)
    return index, values[index]


def render_plot(
    tensor: TensorInfo,
    expert: int,
    weights: np.ndarray,
    order: str,
    max_points: int,
    yscale: str,
    save_path: Path | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    sorted_weights = np.sort(np.abs(weights) if order == "absolute" else weights)
    x, y = decimate_sorted(sorted_weights, max_points)

    # 1 means nonzero, 0 means exact zero. Original order is the meaningful
    # physical-layout test; sorted order is shown only as a comparison.
    original_tree = tree_compression_cost(weights != 0.0)
    sorted_tree = tree_compression_cost(sorted_weights != 0.0)

    print(tree_result_text("zero-mask original order", original_tree), flush=True)
    print(tree_result_text("zero-mask sorted order  ", sorted_tree), flush=True)
    print("Tree cost covers only the zero/nonzero mask; nonzero values are not included.", flush=True)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(x, y, linewidth=0.8)
    original_yes = "YES" if original_tree["effective"] else "NO"
    sorted_yes = "YES" if sorted_tree["effective"] else "NO"
    ax.set_title(
        f"{tensor.name} | {tensor.dtype} {tensor.dims} | expert {expert}\n"
        f"{sparsity_text(weights)}\n"
        f"exact-zero mask tree — original ratio={original_tree['ratio']:.6f} "
        f"effective={original_yes}; sorted ratio={sorted_tree['ratio']:.6f} "
        f"effective={sorted_yes}  (mask only)"
    )
    ax.set_xlabel(f"Sorted weight rank (of {weights.size:,})")
    ax.set_ylabel("Absolute weight" if order == "absolute" else "Weight")
    ax.grid(True, alpha=0.25)
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=1e-5)
    elif yscale == "log":
        if order != "absolute":
            raise ValueError("--yscale log requires --order absolute")
        ax.set_yscale("log")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"saved {save_path}", flush=True)
    if show:
        print("Close the plot window to load the next expert.", flush=True)
        plt.show(block=True)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot sorted GGUF expert weights and evaluate exact-zero-mask tree compression."
    )
    p.add_argument("model", type=Path)
    p.add_argument(
        "--library",
        type=Path,
        default=Path(__file__).resolve().with_name("libgguf_weight_tree_bridge.so"),
    )
    p.add_argument("--list", action="store_true", help="List expert tensors and exit")
    p.add_argument("--tensor", help="Regex selecting tensor names")
    p.add_argument("--layer", type=int, help="Convenience filter for blk.N")
    p.add_argument("--projection", choices=("gate", "up", "down"))
    p.add_argument(
        "--experts",
        default="all",
        help="Examples: all, 0, 0,2,7-10, or 0:256:4",
    )
    p.add_argument("--order", choices=("signed", "absolute"), default="signed")
    p.add_argument("--yscale", choices=("linear", "symlog", "log"), default="symlog")
    p.add_argument(
        "--max-plot-points",
        type=int,
        default=200_000,
        help="Sort all weights but draw at most this many points",
    )
    p.add_argument("--save-dir", type=Path, help="Also save each plot as PNG")
    p.add_argument("--no-show", action="store_true", help="Save/list without opening windows")
    p.add_argument("--limit", type=int, help="Stop after this many plots")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not args.model.is_file():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2
    if not args.library.is_file():
        print(
            f"binding not found: {args.library}\n"
            "Run ./build_gguf_weight_tree_viewer.sh inside the Colibri c directory.",
            file=sys.stderr,
        )
        return 2
    if args.max_plot_points < 100:
        print("--max-plot-points must be at least 100", file=sys.stderr)
        return 2

    with GgufWeights(args.model, args.library) as loader:
        tensors = loader.expert_tensors()
        if not tensors:
            print("no stacked *.ffn_{gate,up,down}_exps.weight tensors found", file=sys.stderr)
            return 3

        if args.list:
            for t in tensors:
                print(
                    f"[{t.index:4d}] {t.name:42s} dtype={t.dtype:5s} "
                    f"dims={t.dims} weights/expert={t.weights_per_expert:,}"
                )
            return 0

        chosen = choose_tensors(tensors, args.tensor, args.layer, args.projection)
        if not chosen:
            print("no expert tensors matched the requested filters", file=sys.stderr)
            return 3
        if not (args.tensor or args.layer is not None or args.projection):
            print(f"No tensor filter given; using first tensor: {chosen[0].name}", flush=True)

        plotted = 0
        for tensor in chosen:
            try:
                experts = parse_expert_spec(args.experts, tensor.experts)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            for expert in experts:
                if args.limit is not None and plotted >= args.limit:
                    return 0
                print(
                    f"loading {tensor.name}, expert {expert}/{tensor.experts - 1}, "
                    f"{tensor.weights_per_expert:,} weights...",
                    flush=True,
                )
                weights = loader.load_expert(tensor, expert)
                print(sparsity_text(weights), flush=True)
                save_path = None
                if args.save_dir:
                    safe_name = tensor.name.replace("/", "_")
                    save_path = args.save_dir / f"{safe_name}.expert-{expert:03d}.png"
                render_plot(
                    tensor,
                    expert,
                    weights,
                    args.order,
                    args.max_plot_points,
                    args.yscale,
                    save_path,
                    not args.no_show,
                )
                plotted += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
