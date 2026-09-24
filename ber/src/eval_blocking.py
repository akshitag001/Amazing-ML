"""Blocking recall ceiling on the validation split (pairs + per-entity + by strategy/k)."""
import sys

import polars as pl

from blocking import candidates_path
from common import is_val, load_truth_pairs


def main():
    cand = pl.read_parquet(candidates_path("train"))
    truth, all_s1 = load_truth_pairs("train")
    val = pl.DataFrame({"s1": all_s1.filter(pl.Series(is_val(all_s1)))})
    cv = cand.join(val, on="s1", how="semi")
    tv = truth.join(val, on="s1", how="semi")
    hit = tv.join(cv, on=["s1", "s23"], how="left").with_columns(pl.col("i1").is_not_null().alias("hit"))
    print(f"val S1 {val.height:,} | candidates {cv.height:,} ({cv.height / val.height:.1f}/S1) | true pairs {tv.height:,}")
    print(f"PAIR RECALL (union): {hit['hit'].mean():.4f}")
    h = hit.filter("hit")
    print(f"  via A (keys):          {(hit['n_keys'].fill_null(0) > 0).mean():.4f}")
    for k in (5, 10, 20, 40):
        print(f"  via B fwd top-{k:<3}:     {(hit['rank_fwd'].fill_null(999) < k).mean():.4f}")
    print(f"  via B reverse top-2:   {hit['rank_rev'].is_not_null().mean():.4f}")
    ent = hit.group_by("s1").agg(pl.col("hit").mean().alias("r"))
    print(f"entity-level: all matches found {(ent['r'] == 1).mean():.4f}; mean entity recall {ent['r'].mean():.4f}")
    return hit


if __name__ == "__main__":
    main()
