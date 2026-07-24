#!/usr/bin/env python3
"""Plan or benchmark Qwen focused hot-expert layers.

The usage file format is Colibri's existing text format:
    layer expert selection_count

`recommend` applies the same greedy policy as the C runtime for K=1..48 and
uses a simple hardware-cost model for resident grouped execution. The usage file
contains marginal counts rather than per-token co-selection records, so grouped
hit probabilities are estimated with a Poisson-binomial independence model.
`benchmark` runs the real model for each K while restoring the exact same usage
history before every run, then ranks configurations by measured tok/s.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Iterable

TOK_RE = re.compile(r"\(([0-9]+(?:\.[0-9]+)?) tok/s\)")
SCHED_RE = re.compile(r"\[SCHED\] hits=(\d+).*?misses=(\d+)")
FOCUS_RE = re.compile(r"\[FOCUS\].*?effective=(\d+).*?layers=([^\n]+)")


def load_usage(path: Path, layers: int, experts: int) -> list[list[int]]:
    usage = [[0] * experts for _ in range(layers)]
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(f"{path}:{lineno}: expected 'layer expert count'")
            layer, expert, count = map(int, parts)
            if 0 <= layer < layers and 0 <= expert < experts and count >= 0:
                usage[layer][expert] += count
    return usage


def focus_plan(usage: list[list[int]], pin_slots: int, k: int) -> tuple[list[int], list[tuple[int, int, int]], int]:
    layers = len(usage)
    experts = len(usage[0]) if layers else 0
    if pin_slots <= 0 or k <= 0 or not layers or not experts:
        return [], [], 0
    k = min(k, layers, pin_slots)
    layer_cap = (pin_slots + k - 1) // k
    ranked = [sorted(enumerate(row), key=lambda x: (-x[1], x[0])) for row in usage]
    scores = []
    for layer, row in enumerate(ranked):
        score = sum(count for _, count in row[:layer_cap])
        scores.append((score, layer))
    selected = [layer for score, layer in sorted(scores, key=lambda x: (-x[0], x[1])) if score > 0][:k]
    if not selected:
        return [], [], 0

    chosen: list[tuple[int, int, int]] = []
    counts = {layer: 0 for layer in selected}
    # One expert per chosen layer makes K a real concentration parameter.
    for layer in sorted(selected):
        expert, count = ranked[layer][0]
        if count > 0 and len(chosen) < pin_slots:
            chosen.append((layer, expert, count))
            counts[layer] += 1
    candidates = []
    for layer in selected:
        for expert, count in ranked[layer][1:layer_cap]:
            if count > 0:
                candidates.append((count, layer, expert))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    for count, layer, expert in candidates:
        if len(chosen) >= pin_slots:
            break
        if counts[layer] >= layer_cap:
            continue
        chosen.append((layer, expert, count))
        counts[layer] += 1
    covered = sum(count for _, _, count in chosen)
    selected_sorted = sorted(selected, key=lambda layer: (-sum(c for _, c in ranked[layer][:layer_cap]), layer))
    return selected_sorted, chosen, covered



def grouped_cost_model(usage: list[list[int]], chosen: list[tuple[int, int, int]],
                       top_k: int, upload_us: float, launch_us: float,
                       individual_kernels: int, group_kernels: int,
                       group_min: int) -> tuple[float, float, float]:
    """Return expected hits, P(group), and saved microseconds per generated token.

    The history records only marginal selection counts. For each layer, the
    inclusion probability of a pinned expert is count / routed_tokens, and the
    number of simultaneous pinned hits is approximated as a Poisson-binomial
    distribution. Upload avoidance applies to every hit. Group launch savings
    apply only when at least `group_min` resident experts are selected.
    """
    by_layer: dict[int, list[int]] = {}
    for layer, _expert, count in chosen:
        by_layer.setdefault(layer, []).append(count)
    expected_hits = 0.0
    group_probability = 0.0
    saved_us = 0.0
    for layer, counts in by_layer.items():
        total = sum(usage[layer])
        routed_tokens = total / top_k if top_k > 0 else 0.0
        if routed_tokens <= 0:
            continue
        probs = [min(1.0, max(0.0, count / routed_tokens)) for count in counts]
        dist = [1.0]
        for prob in probs:
            nxt = [0.0] * (len(dist) + 1)
            for hits, mass in enumerate(dist):
                nxt[hits] += mass * (1.0 - prob)
                nxt[hits + 1] += mass * prob
            dist = nxt
        eh = sum(hits * mass for hits, mass in enumerate(dist))
        pg = sum(dist[group_min:]) if group_min < len(dist) else 0.0
        launch_saving = 0.0
        for hits in range(group_min, len(dist)):
            launch_saving += dist[hits] * max(0, individual_kernels * hits - group_kernels)
        expected_hits += eh
        group_probability += pg
        saved_us += upload_us * eh + launch_us * launch_saving
    return expected_hits, group_probability, saved_us

def cmd_recommend(args: argparse.Namespace) -> int:
    usage = load_usage(args.usage, args.layers, args.experts)
    total = sum(map(sum, usage))
    rows = []
    max_k = min(args.max_layers, args.layers, args.pin_slots)
    for k in range(1, max_k + 1):
        selected, chosen, covered = focus_plan(usage, args.pin_slots, k)
        coverage = covered / total if total else 0.0
        expected_hits, group_probability, saved_us = grouped_cost_model(
            usage, chosen, args.top_k, args.upload_us, args.launch_us,
            args.individual_kernels, args.group_kernels, args.group_min)
        rows.append((k, len(selected), covered, coverage, selected, chosen,
                     expected_hits, group_probability, saved_us))
    if not rows:
        print("No usable records or pin slots.", file=sys.stderr)
        return 2
    best = max(rows, key=lambda r: (r[8], r[2], -r[0]))
    print(f"usage={args.usage} total_selections={total} pin_slots={args.pin_slots}")
    print(f"model: upload={args.upload_us:g}us launch={args.launch_us:g}us "
          f"individual={args.individual_kernels} kernels group={args.group_kernels} kernels min={args.group_min}")
    if args.show_layers:
        print(f"{'K':>3} {'used':>4} {'coverage':>10} {'hits/tok':>9} {'groups/tok':>10} {'save us/tok':>11} layers")
    else:
        print(f"{'K':>3} {'coverage':>10} {'hits/tok':>9} {'groups/tok':>10} {'save us/tok':>11}")
    for k, used, covered, coverage, selected, _, expected_hits, group_probability, saved_us in rows:
        mark = " *" if k == best[0] else ""
        if args.show_layers:
            layers_text = ",".join(map(str, selected))
            print(f"{k:3d} {used:4d} {coverage:9.3%} {expected_hits:9.3f} "
                  f"{group_probability:10.3f} {saved_us:11.1f} {layers_text}{mark}")
        else:
            print(f"{k:3d} {coverage:9.3%} {expected_hits:9.3f} "
                  f"{group_probability:10.3f} {saved_us:11.1f}{mark}")
    print()
    print(f"Recommended focused-layer count: {best[0]}")
    print(f"Estimated saving at K={best[0]}: {best[8]:.1f} us/token")
    print(f"Recorded-selection coverage at K={best[0]}: {best[3]:.3%}")
    print("This is a marginal-history model; confirm the best nearby K values with measured tok/s.")
    return 0


def extract_metrics(text: str) -> tuple[float | None, int | None, int | None, str]:
    speeds = [float(x) for x in TOK_RE.findall(text)]
    speed = speeds[-1] if speeds else None
    sched = SCHED_RE.findall(text)
    hits = int(sched[-1][0]) if sched else None
    misses = int(sched[-1][1]) if sched else None
    focus = FOCUS_RE.findall(text)
    layers = focus[-1][1].strip() if focus else ""
    return speed, hits, misses, layers


def cmd_benchmark(args: argparse.Namespace) -> int:
    binary = args.binary.resolve()
    model = args.model.resolve()
    usage = args.usage.resolve()
    if not binary.exists():
        raise FileNotFoundError(binary)
    if not model.exists():
        raise FileNotFoundError(model)
    if not usage.exists():
        raise FileNotFoundError(usage)
    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    original = usage.read_bytes()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = args.output or Path.cwd() / f"qwen-focus-bench-{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results.csv"
    results: list[dict[str, object]] = []
    ks: Iterable[int] = range(args.start, min(args.end, args.layers) + 1)
    if args.include_baseline:
        ks = [0, *ks]
    try:
        for round_no in range(1, args.rounds + 1):
            for k in ks:
                usage.write_bytes(original)
                env = os.environ.copy()
                env.update({
                    "QWEN_ROUTER_GPU": str(args.router_gpu),
                    "PILOT": "0",
                    "PILOT_REAL": "0",
                    "PIN": "auto",
                    "AUTOPIN": "1",
                    "CUDA_RESERVE_GB": str(args.reserve_gb),
                    "QWEN_FOCUS_LAYER_COUNT": str(k),
                })
                if args.pin_gb is not None:
                    env["PIN_GB"] = str(args.pin_gb)
                else:
                    env.pop("PIN_GB", None)
                label = "baseline" if k == 0 else f"k{k:02d}"
                log_path = outdir / f"r{round_no}-{label}.log"
                cmd = [str(binary), "--gguf", str(model), "--device", args.device,
                       "--prompt", prompt, "--max-tokens", str(args.max_tokens), "--verbose"]
                print(f"===== round {round_no}: K={k} =====", flush=True)
                proc = subprocess.run(cmd, cwd=binary.parent, env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, errors="replace")
                log_path.write_text(proc.stdout, encoding="utf-8")
                speed, hits, misses, focused = extract_metrics(proc.stdout)
                row = {"round": round_no, "k": k, "tok_s": speed or 0.0,
                       "hits": hits if hits is not None else -1,
                       "misses": misses if misses is not None else -1,
                       "focused_layers": focused, "returncode": proc.returncode,
                       "log": str(log_path)}
                results.append(row)
                print(f"return={proc.returncode} tok/s={speed} hits={hits} misses={misses} layers={focused}")
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader(); writer.writerows(results)
    finally:
        usage.write_bytes(original)

    grouped: dict[int, list[float]] = {}
    for row in results:
        if row["returncode"] == 0 and float(row["tok_s"]) > 0:
            grouped.setdefault(int(row["k"]), []).append(float(row["tok_s"]))
    ranking = sorted(((statistics.median(v), statistics.mean(v), k, len(v))
                      for k, v in grouped.items()), reverse=True)
    print("\n===== RANKING =====")
    print(f"{'rank':>4} {'K':>4} {'median':>9} {'mean':>9} {'runs':>5}")
    for rank_no, (median, mean, k, count) in enumerate(ranking, 1):
        print(f"{rank_no:4d} {k:4d} {median:9.3f} {mean:9.3f} {count:5d}")
    if ranking:
        print(f"\nEmpirically fastest K={ranking[0][2]} at median {ranking[0][0]:.3f} tok/s.")
    print(f"Results: {csv_path}")
    print(f"Logs: {outdir}")
    print(f"Original usage history restored: {usage}")
    return 0 if ranking else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("recommend", help="estimate K from recorded expert selections")
    r.add_argument("usage", type=Path)
    r.add_argument("--pin-slots", type=int, required=True,
                   help="number of hot pin slots (for example the [PIN] count from a normal run)")
    r.add_argument("--layers", type=int, default=48)
    r.add_argument("--experts", type=int, default=512)
    r.add_argument("--max-layers", type=int, default=48)
    r.add_argument("--top-k", type=int, default=10,
                   help="routed experts selected per layer and token")
    r.add_argument("--upload-us", type=float, default=2000.0,
                   help="estimated cost avoided by a resident expert hit")
    r.add_argument("--launch-us", type=float, default=20.0,
                   help="estimated CUDA launch/synchronization cost")
    r.add_argument("--individual-kernels", type=int, default=5,
                   help="kernel-equivalent launches for one sequential expert")
    r.add_argument("--group-kernels", type=int, default=7,
                   help="fixed kernel-equivalent launches for one resident group")
    r.add_argument("--group-min", type=int, default=2,
                   help="minimum simultaneous resident hits needed to group")
    r.add_argument("--show-layers", action="store_true",
                   help="also print the internally selected layer IDs for diagnostics")
    r.set_defaults(func=cmd_recommend)

    b = sub.add_parser("benchmark", help="measure K=1..48 using the real model")
    b.add_argument("usage", type=Path)
    b.add_argument("--model", type=Path, required=True)
    b.add_argument("--binary", type=Path, default=Path("./colibri"))
    b.add_argument("--prompt", default="Tell a short story about France.")
    b.add_argument("--prompt-file", type=Path)
    b.add_argument("--max-tokens", type=int, default=60)
    b.add_argument("--rounds", type=int, default=1)
    b.add_argument("--start", type=int, default=1)
    b.add_argument("--end", type=int, default=48)
    b.add_argument("--layers", type=int, default=48)
    b.add_argument("--reserve-gb", type=float, default=3.5)
    b.add_argument("--pin-gb", type=float)
    b.add_argument("--router-gpu", type=int, choices=(0, 1), default=0)
    b.add_argument("--device", default="cuda")
    b.add_argument("--include-baseline", action="store_true")
    b.add_argument("--output", type=Path)
    b.set_defaults(func=cmd_benchmark)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
