"""Decision support before spending a leaderboard upload.

1. Paired bootstrap: per-entity F0.5 of the new model vs the reference model on the SAME validation S1
   entities; resample entities to get a confidence interval for the improvement.
2. Density robustness (distractor-density check): F0.5 per bucket of candidates-per-S1, and the expected
   F0.5 after reweighting validation buckets to the TEST candidate-density distribution.

Both models are compared on raw thresholded predictions (no conflict resolution) so they are strictly paired.

usage: python src/compare_val.py <ref_val_pred.parquet> <ref_thr> <new_val_pred.parquet> <new_thr>
"""
import sys

import numpy as np
import polars as pl

from common import is_val, load_truth_pairs

BUCKETS = [0, 10, 20, 30, 40, 60, 80, 120, 10**9]


def per_entity_f(pred, thr, truth, s1_ids):
    s1 = pl.DataFrame({"s1": s1_ids})
    p = pred.filter(pl.col("p") >= thr).select("s1", "s23").join(s1, on="s1", how="semi")
    tp = p.join(truth, on=["s1", "s23"]).group_by("s1").len("tp")
    df = (s1.join(tp, on="s1", how="left").join(p.group_by("s1").len("np"), on="s1", how="left")
          .join(truth.group_by("s1").len("nt"), on="s1", how="left").fill_null(0))
    tp_, np_, nt_ = (df[c].to_numpy().astype(float) for c in ("tp", "np", "nt"))
    P = np.divide(tp_, np_, out=np.zeros_like(tp_), where=np_ > 0)
    R = np.divide(tp_, nt_, out=np.zeros_like(tp_), where=nt_ > 0)
    d = 0.25 * P + R
    F = np.divide(1.25 * P * R, d, out=np.zeros_like(P), where=d > 0)
    F[nt_ == 0] = (np_[nt_ == 0] == 0)
    return df.select("s1").with_columns(pl.Series("F", F))


def bucket(col):
    return pl.col(col).cut(BUCKETS[1:-1], left_closed=True).alias("bucket")


def main(ref_path, ref_thr, new_path, new_thr):
    truth, all_s1 = load_truth_pairs("train")
    val_ids = all_s1.filter(pl.Series(is_val(all_s1)))
    truth = truth.join(pl.DataFrame({"s1": val_ids}), on="s1", how="semi")
    a = per_entity_f(pl.read_parquet(ref_path), float(ref_thr), truth, val_ids)
    b = per_entity_f(pl.read_parquet(new_path), float(new_thr), truth, val_ids)
    d = a.join(b, on="s1", suffix="_new")
    fa, fb = d["F"].to_numpy(), d["F_new"].to_numpy()
    diff = fb - fa
    rng = np.random.default_rng(20260925)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"validation entities {n:,}")
    print(f"  reference F0.5 {fa.mean():.5f} | new F0.5 {fb.mean():.5f} | improvement {diff.mean():+.5f}")
    print(f"  95% bootstrap CI of improvement: [{lo:+.5f}, {hi:+.5f}]  | P(improvement <= 0) = "
          f"{(boots <= 0).mean():.4f}")
    print(f"  entities better {int((diff > 1e-9).sum()):,} | worse {int((diff < -1e-9).sum()):,} | unchanged "
          f"{int((np.abs(diff) <= 1e-9).sum()):,}")

    # density robustness: validation F by candidates-per-S1 bucket, reweighted to the test distribution
    def density(mode, ids=None):
        c = pl.scan_parquet(f"cache/{mode}_candidates.parquet").group_by("s1").len("ncand").collect()
        return c.join(pl.DataFrame({"s1": ids}), on="s1", how="right").fill_null(0) if ids is not None else c
    dv = d.join(density("train", val_ids), on="s1").with_columns(bucket("ncand"))
    te = density("test").with_columns(bucket("ncand"))
    wt = te.group_by("bucket").len("n_test").with_columns(pl.col("n_test") / pl.col("n_test").sum())
    g = (dv.group_by("bucket").agg(pl.len().alias("n_val"), pl.col("F").mean().alias("F_ref"),
                                   pl.col("F_new").mean().alias("F_new"))
         .with_columns(pl.col("n_val") / pl.col("n_val").sum()).join(wt, on="bucket", how="full", coalesce=True)
         .fill_null(0).sort("bucket"))
    with pl.Config(tbl_rows=20, float_precision=4):
        print("\nF0.5 by candidates-per-S1 bucket (share of validation vs test entities):")
        print(g.rename({"n_val": "share_val", "n_test": "share_test"}))
    for col in ("F_ref", "F_new"):
        print(f"  {col}: validation {float((g['n_val'] * g[col]).sum()):.5f} -> reweighted to test density "
              f"{float((g['n_test'] * g[col]).sum()):.5f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
