"""Dataset IO: Parquet when pyarrow is available, gzipped CSV otherwise; plus content hashing.

Layout:  <out>/dims|observed|latent/<table>.<parquet|csv.gz>  and  <out>/manifest.json
The manifest records config digest, seed, day range, row counts and a content hash per table, so a
nightly job can verify that extending history left the existing prefix untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .schema import ALL_TABLES, DAY_INDEXED, DIMS, LATENT, OBSERVED, coerce
from .simulate import SimResult


def have_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def frame_hash(df: pd.DataFrame) -> str:
    """Order-sensitive content hash (values, column names and dtypes)."""
    h = hashlib.sha256()
    h.update("|".join(f"{c}:{df[c].dtype}" for c in df.columns).encode())
    if len(df):
        h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def table_hashes(tables: dict[str, pd.DataFrame], up_to_day: int | None = None) -> dict[str, str]:
    """Hash every table; with up_to_day, day-indexed tables are restricted to day_index < up_to_day."""
    out = {}
    for name, df in tables.items():
        if up_to_day is not None and name in DAY_INDEXED:
            df = df[df["day_index"] < up_to_day]
        out[name] = frame_hash(df.reset_index(drop=True))
    return out


def _group(name: str) -> str:
    return "dims" if name in DIMS else "observed" if name in OBSERVED else "latent"


def write_dataset(res: SimResult, out_dir: str | Path, fmt: str = "auto", include_latent: bool = True) -> dict:
    fmt = ("parquet" if have_pyarrow() else "csv.gz") if fmt == "auto" else fmt
    if fmt == "parquet" and not have_pyarrow():
        raise RuntimeError("pyarrow is not installed; use fmt='csv.gz' or install pyarrow")
    out = Path(out_dir)
    entries = {}
    for name, df in res.tables.items():
        if not include_latent and name in LATENT:
            continue
        path = out / _group(name) / f"{name}.{fmt}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False, compression="gzip")
        entries[name] = {"file": str(path.relative_to(out)), "rows": int(len(df)), "sha256": frame_hash(df)}
    manifest = {"format": fmt, "config_digest": res.cfg.digest(), "seed": res.cfg.seed, "n_days": res.n_days,
                "start_date": res.cfg.start_date, "latent_included": include_latent, "tables": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read_table(out_dir: str | Path, name: str) -> pd.DataFrame:
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    path = out / manifest["tables"][name]["file"]
    if manifest["format"] == "parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
    return coerce(df, name)
