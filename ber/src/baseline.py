"""Checkpoint-1 safety-net baseline: exact normalized-name + country match.

usage: python src/baseline.py train   # evaluates on the validation split
       python src/baseline.py test    # writes baseline submission for test
"""
import sys
import time

import polars as pl
from anyascii import anyascii

from common import OUT_DIR, f05_eval, is_val, load_sources, load_truth_pairs, write_submission


def key(df):
    names = [anyascii(x or "") for x in df["business_name"].to_list()]
    return df.with_columns(
        pl.Series("k", names).str.to_lowercase().str.replace_all(r"[^a-z0-9]", "")
    ).filter(pl.col("k") != "")


def main(mode):
    t = time.time()
    s1, s23 = load_sources(mode)
    k1, k23 = key(s1), key(s23)
    # only S1 keys that are unique within their country (ambiguous keys -> no match)
    k1 = k1.filter(pl.len().over("k", "country") == 1)
    pairs = (k1.select(pl.col("entity_id").alias("s1"), "k", "country")
             .join(k23.select(pl.col("entity_id").alias("s23"), "k", "country"), on=["k", "country"]))
    print(f"{len(pairs):,} pairs in {time.time() - t:.0f}s")
    if mode == "train":
        truth, all_s1 = load_truth_pairs(mode)
        val_ids = all_s1.filter(pl.Series(is_val(all_s1)))
        print("baseline on validation split:")
        f05_eval(pairs, truth, val_ids)
        write_submission(val_ids, pairs, pairs, OUT_DIR, prefix="val_baseline_")
    else:
        write_submission(s1["entity_id"], pairs, pairs, OUT_DIR, prefix="baseline_")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
