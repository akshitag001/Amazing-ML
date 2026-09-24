"""Feature sanity check: per-feature stats on train and test, zero-variance / high-null flags,
and a train-vs-test schema + distribution comparison (same code path => same columns, similar ranges).

usage: python src/feature_check.py
"""
import glob
import os

import numpy as np
import polars as pl

from features import feat_dir

META = {"i1", "i23", "s1", "s23", "y", "is_val"}
NULL_FLAG = 0.30  # flag features missing on more than 30% of pairs


def stats(mode):
    lf = pl.scan_parquet(sorted(glob.glob(os.path.join(feat_dir(mode), "part_*.parquet"))))
    cols = [c for c in lf.collect_schema().names() if c not in META]
    agg = []
    for c in cols:
        x = pl.col(c).fill_nan(None)
        agg += [x.min().alias(f"{c}|min"), x.max().alias(f"{c}|max"), x.mean().alias(f"{c}|mean"),
                x.std().alias(f"{c}|std"), x.is_null().mean().alias(f"{c}|null")]
    r = lf.select(pl.len().alias("n_pairs"), *agg).collect()
    rows = [{"feature": c, **{k: r[f"{c}|{k}"][0] for k in ("min", "max", "mean", "std", "null")}} for c in cols]
    return r["n_pairs"][0], pl.DataFrame(rows), cols


def main():
    ntr, tr, ctr = stats("train")
    nte, te, cte = stats("test")
    print(f"train pairs {ntr:,} | test pairs {nte:,}")
    print(f"schema identical: {ctr == cte}  (train-only columns: {set(ctr) - set(cte)}, "
          f"test-only: {set(cte) - set(ctr)})")
    m = tr.join(te, on="feature", suffix="_test")
    m = m.with_columns(
        pl.when(pl.col("std").fill_null(0) == 0).then(pl.lit("ZERO-VAR"))
        .when(pl.col("null") > NULL_FLAG).then(pl.lit("HIGH-NULL")).otherwise(pl.lit("")).alias("flag"),
        ((pl.col("mean_test") - pl.col("mean")) / pl.col("std").clip(1e-9)).alias("shift_sd"),
    )
    with pl.Config(tbl_rows=200, tbl_width_chars=200, float_precision=3):
        print(m.select("feature", "min", "max", "mean", "std", "null", "mean_test", "null_test", "shift_sd", "flag"))
    print("\nflagged:", m.filter(pl.col("flag") != "").select("feature", "flag", "null").rows())
    big = m.filter(pl.col("shift_sd").abs() > 0.5).select("feature", "mean", "mean_test", "shift_sd")
    print("train->test mean shift > 0.5 sd:", big.rows())
    y = pl.scan_parquet(sorted(glob.glob(os.path.join(feat_dir("train"), "part_*.parquet")))).select(
        pl.col("y").sum().alias("pos"), pl.len().alias("n"), pl.col("is_val").sum().alias("val_pairs")).collect()
    print("train labels:", y.rows(named=True))


if __name__ == "__main__":
    main()
