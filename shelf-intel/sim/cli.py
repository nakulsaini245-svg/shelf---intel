"""Command line:  python -m sim.cli generate|check [--config PATH] [--days N] [--out DIR] [--seed S]"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .io import have_pyarrow, write_dataset
from .simulate import simulate
from .validate import validate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sim", description="Synthetic quick-commerce world generator")
    ap.add_argument("command", choices=["generate", "check"])
    ap.add_argument("--config", default="configs/world_v1.yaml")
    ap.add_argument("--days", type=int, default=None, help="override n_days (e.g. 5 for a smoke test)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="data/world_v1")
    ap.add_argument("--fmt", default="auto", choices=["auto", "parquet", "csv.gz"])
    ap.add_argument("--no-latent", action="store_true", help="omit latent truth tables from the written dataset")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    if a.seed is not None:
        cfg = replace(cfg, seed=a.seed)
    t0 = time.time()
    res = simulate(cfg, n_days=a.days)
    print(f"simulated {res.n_days} days in {time.time() - t0:.1f}s "
          f"({sum(len(t) for t in res.tables.values()):,} rows, config {cfg.digest()})")
    checks = validate(res, include_targets=True)
    for c in checks:
        if not c.ok or a.command == "check":
            print(c)
    bad_inv = [c for c in checks if not c.ok and not c.name.startswith("target:")]
    bad_tgt = [c for c in checks if not c.ok and c.name.startswith("target:")]
    print(f"invariants: {len(checks) - len([c for c in checks if c.name.startswith('target:')]) - len(bad_inv)} pass, "
          f"{len(bad_inv)} fail | targets: {len(bad_tgt)} out of range")
    if a.command == "generate":
        fmt = a.fmt
        if fmt == "auto" and not have_pyarrow():
            print("note: pyarrow not installed, writing gzipped CSV instead of Parquet")
        m = write_dataset(res, Path(a.out), fmt=fmt, include_latent=not a.no_latent)
        print(f"wrote {len(m['tables'])} tables ({m['format']}) to {a.out}")
    return 1 if bad_inv else 0


if __name__ == "__main__":
    sys.exit(main())
