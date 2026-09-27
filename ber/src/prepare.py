"""Stage 1: normalize all records of a mode and cache them as parquet.

After normalization, missing regions are imputed from the locality (city) using a locality -> region map
learned from the SAME dataset's Source-1 records (which always carry a region). Some locales write addresses
that stop at the city (seen in the unseen test country); the matcher was trained on data where a present
address always had a region, so filling it in keeps test records in-distribution. Country is only used as
part of the lookup key (a city name can exist in several countries), never for control flow.

usage: python src/prepare.py train|test
"""
import os
import sys
import time

import polars as pl

from common import CACHE_DIR, load_sources
from normalize import normalize_df

REGION_MAP_MIN_COUNT = 3       # locality must be seen with a region at least this often in Source 1
REGION_MAP_MIN_PURITY = 0.90   # ... and map to one region in at least this share of cases


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


def _loc_rows(df):
    return (df.select("entity_id", "country", "addr_region", pl.col("addr_locality").str.split("|").alias("loc"))
            .explode("loc").filter(pl.col("loc").is_not_null() & (pl.col("loc") != "")))


def region_map(s1):
    """(country, locality) -> region, learned from Source-1 records that have both."""
    x = _loc_rows(s1).filter(pl.col("addr_region") != "")
    c = x.group_by("country", "loc", "addr_region").len("n")
    tot = c.group_by("country", "loc").agg(pl.col("n").sum().alias("tot"))
    best = (c.sort("n", descending=True).group_by("country", "loc", maintain_order=True).first()
            .join(tot, on=["country", "loc"]))
    return best.filter((pl.col("tot") >= REGION_MAP_MIN_COUNT) & (pl.col("n") / pl.col("tot") >= REGION_MAP_MIN_PURITY)
                       ).select("country", "loc", pl.col("addr_region").alias("imp_region"), "n")


def impute_regions(df, rmap):
    """Fill addr_region for non-blank addresses that have a known locality but no region."""
    need = df.filter((pl.col("addr_region") == "") & ~pl.col("addr_blank"))
    if need.height == 0:
        return df, 0
    hit = (_loc_rows(need).join(rmap, on=["country", "loc"]).sort("n", descending=True)
           .group_by("entity_id", maintain_order=True).first().select("entity_id", "imp_region"))
    out = (df.join(hit, on="entity_id", how="left", maintain_order="left")
           .with_columns(pl.coalesce(pl.when(pl.col("addr_region") != "").then(pl.col("addr_region")),
                                     pl.col("imp_region"), pl.lit("")).alias("addr_region"))
           .drop("imp_region"))
    return out, hit.height


def main(mode):
    os.makedirs(CACHE_DIR, exist_ok=True)
    s1, s23 = load_sources(mode)
    out = {}
    for name, df in (("s1", s1), ("s23", s23)):
        t = time.time()
        out[name] = normalize_df(df)
        print(f"{mode} {name}: {out[name].height:,} rows normalized in {time.time() - t:.0f}s")
    rmap = region_map(out["s1"])
    for name in ("s1", "s23"):
        out[name], n = impute_regions(out[name], rmap)
        out[name].write_parquet(cache_path(mode, name))
        print(f"{mode} {name}: region imputed from locality for {n:,} records ({len(rmap):,} locality keys)")


if __name__ == "__main__":
    main(sys.argv[1])
