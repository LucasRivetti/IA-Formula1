#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-platform batch runner for src.scenario_track
- Works on Windows (PowerShell/CMD) and Linux/macOS
- Replaces the bash+awk pipeline: runs multiple --stintage_grid ranges,
  parses stdout from src.scenario_track, and writes a CSV.

Usage (inside your venv, from repo root):
    python scripts/run_scenarios.py --gp Monza \
        --model models/best_model.joblib \
        --data data/processed.parquet \
        --output results/res_data.csv \
        --top 5

If you keep the default paths/values, just:
    python scripts/run_scenarios.py

Tip: Ensure your virtualenv is activated so `python -m src.scenario_track` finds deps.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

DEFAULT_COMMANDS = [
    "--stintage_grid 10 20 1 --compounds SOFT,MEDIUM",
    "--stintage_grid 20 30 1 --compounds MEDIUM,HARD",
    "--stintage_grid 25 35 1 --compounds SOFT,HARD",
    "--stintage_grid 30 40 1 --compounds MEDIUM,HARD",
    "--stintage_grid 5 25 1 --compounds SOFT",
    "--stintage_grid 15 45 1 --compounds MEDIUM",
    "--stintage_grid 25 53 1 --compounds HARD",
    "--stintage_grid 15 20 1 --compounds SOFT,MEDIUM",
    "--stintage_grid 17 22 1 --compounds MEDIUM,HARD",
    "--stintage_grid 15 25 1 --compounds SOFT,HARD",
    "--stintage_grid 18 18 1 --compounds SOFT,MEDIUM,HARD",
    "--stintage_grid 25 25 1 --compounds SOFT,MEDIUM,HARD",
    "--stintage_grid 32 32 1 --compounds SOFT,MEDIUM,HARD",
    "--stintage_grid 5 53 1 --compounds SOFT,MEDIUM,HARD",
    "--stintage_grid 10 50 1 --compounds SOFT,MEDIUM,HARD",
    "--stintage_grid 15 45 1 --compounds MEDIUM,HARD",
    "--stintage_grid 10 30 1 --compounds SOFT,MEDIUM",
    "--stintage_grid 20 40 1 --compounds MEDIUM,HARD",
    "--stintage_grid 15 40 1 --compounds SOFT,HARD",
    "--stintage_grid 0 53 1 --compounds SOFT,MEDIUM,HARD",
]

SKIP_PATTERNS = (
    r"^=+",              # ====== separators
    r"cenário",          # Portuguese words in headers
    r"recomendação",
    r"^Top",             # Top-N header
    r"^geral",
)

NUM_RE = re.compile(r"^[0-9]+(?:[.,][0-9]+)?$")

def parse_stdout_to_rows(stdout: str) -> List[Tuple[str, str, str]]:
    """
    Parse stdout emitted by src.scenario_track into rows:
        (compound, stintage, gap_%_melhor_volta)

    Heuristic mirrors the original awk:
    - drop lines that match skip patterns
    - collapse whitespace
    - keep lines with exactly 3 fields
    - fields[1] and fields[2] must look like numbers
    """
    rows: List[Tuple[str, str, str]] = []
    skip_res = [re.compile(pat, flags=re.IGNORECASE) for pat in SKIP_PATTERNS]

    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(s.search(line) for s in skip_res):
            continue
        # collapse multiple whitespace to single space
        line = re.sub(r"\s+", " ", line)
        parts = line.split(" ")
        if len(parts) == 3:
            cpd, stin, gap = parts
            if NUM_RE.match(stin) and NUM_RE.match(gap):
                rows.append((cpd, stin, gap))
    return rows


def run_one(command_tail: List[str], model: Path, data: Path, gp: str, top: int) -> List[Tuple[str, str, str]]:
    """
    Execute: python -u -m src.scenario_track --model MODEL --data DATA --gp GP <command_tail...> --top TOP
    Returns parsed rows.
    """
    py = sys.executable  # current interpreter (works in venv on Win/Linux)
    base_cmd = [
        py, "-u", "-m", "src.scenario_track",
        "--model", str(model),
        "--data", str(data),
        "--gp", gp,
        "--top", str(top),
    ]
    full_cmd = base_cmd + command_tail

    env = os.environ.copy()
    # keep BLAS threads small for stable perf/logs
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    proc = subprocess.run(full_cmd, capture_output=True, text=True, env=env, errors="ignore")
    if proc.returncode != 0:
        sys.stderr.write(f"[warn] scenario_track returned non-zero code={proc.returncode}\n")
        if proc.stderr:
            sys.stderr.write(proc.stderr + "\n")
    out = proc.stdout or ""
    return parse_stdout_to_rows(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Batch scenarios for src.scenario_track → CSV")
    p.add_argument("--gp", default="Monza", help="GP key or substring (e.g., Monza, Interlagos, Spa)")
    p.add_argument("--model", default="models/best_model.joblib", type=Path)
    p.add_argument("--data", default="data/processed.parquet", type=Path)
    p.add_argument("--output", default="results/res_data.csv", type=Path)
    p.add_argument("--top", default=5, type=int, help="Top-N combinations to capture from each run")
    p.add_argument("--commands_json", type=Path, help="Optional JSON file with a list of command strings")
    p.add_argument("--dry_run", action="store_true", help="Only print commands; do not execute")
    args = p.parse_args(argv)

    # load custom commands if provided
    if args.commands_json and args.commands_json.exists():
        try:
            import json as _json
            cmds = list(map(str, _json.loads(args.commands_json.read_text(encoding='utf-8'))))
        except Exception as e:
            sys.stderr.write(f"[warn] Failed to parse commands_json, using defaults. Error: {e}\n")
            cmds = DEFAULT_COMMANDS
    else:
        cmds = DEFAULT_COMMANDS

    # ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # write CSV header
    with args.output.open("w", encoding="utf-8", newline="") as f:
        f.write("compound,stintage,gap_%_melhor_volta\n")

    total = len(cmds)
    for i, cmd_str in enumerate(cmds, start=1):
        # split respecting quotes, but our commands are simple → just .split() is fine
        tail = cmd_str.split()
        sys.stderr.write(f">> [{i}/{total}] Running: {cmd_str}\n")
        if args.dry_run:
            continue
        rows = run_one(tail, args.model, args.data, args.gp, args.top)
        # append to CSV
        if rows:
            with args.output.open("a", encoding="utf-8", newline="") as f:
                for cpd, stin, gap in rows:
                    # normalize decimal separator to dot for CSV numeric parsing
                    stin_norm = stin.replace(",", ".")
                    gap_norm = gap.replace(",", ".")
                    f.write(f"{cpd},{stin_norm},{gap_norm}\n")
        else:
            sys.stderr.write("[info] No rows parsed from this run.\n")

    print(f"✅ Done. CSV at: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
