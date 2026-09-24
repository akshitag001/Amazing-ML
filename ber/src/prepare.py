"""Stage 1: normalize all records of a mode and cache them as parquet.

usage: python src/prepare.py train|test
"""
import os
import sys
import time

import polars as pl

from common import CACHE_DIR, load_sources
from normalize import normalize_df


def cache_path(mode, name):
    return os.path.join(CACHE_DIR, f"{mode}_{name}.parquet")


def load_norm(mode, columns=None):
    return (pl.read_parquet(cache_path(mode, "s1"), columns=columns),
            pl.read_parquet(cache_path(mode, "s23"), columns=columns))


def load_sorted(mode, columns):
    """Normalized records sorted by (country, entity_id): the canonical row order that blocking indices
    (i1 / i23) refer to. Deterministic because entity_id is unique."""
    cols = list(dict.fromkeys(["entity_id", "country"] + columns))
    return tuple(d.sort("country", "entity_id") for d in load_norm(mode, cols))


def main(mode):
    os.makedirs(CACHE_DIR, exist_ok=True)
    s1, s23 = load_sources(mode)
    for name, df in (("s1", s1), ("s23", s23)):
        t = time.time()
        out = normalize_df(df)
        out.write_parquet(cache_path(mode, name))
        print(f"{mode} {name}: {out.height:,} rows normalized in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
