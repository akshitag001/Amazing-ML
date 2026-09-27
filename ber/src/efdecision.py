"""Expected-F0.5-optimal match selection per S1 entity (replaces a single global threshold).

Per entity, F0.5 = 1.25 * TP / (0.25 * T + k)  (k = #predicted, T = #true). Hence an extra match only pays off
when it is ~80% likely correct, while the FIRST match of an entity pays off from ~46% (an empty prediction
scores 0 unless the entity truly has no match). A global threshold cannot express that; this rule can.

1. Calibrate scores p -> q = P(match) with isotonic regression (fit on validation).
2. Conflict resolution first: each S2/S3 record keeps only its best-scoring S1 claimant.
3. Per S1, candidates sorted by q: expected F for predicting the top-k,
       G_k = 1.25 * sum_{i<=k} q_i / (0.25 * (sum_all q + C) + k)          k >= 1
       G_0 = prod_i (1 - q_i) * exp(-C)                                     (P(entity truly has no match))
   (C = expected true matches outside the candidate set, i.e. blocking misses.) Pick argmax_k.
Country is never read: the rule is per entity, driven only by calibrated probabilities.

Evaluation is cross-fitted: validation S1 entities are split in two halves by hash; calibration and C are fit
on one half and scored on the other, then swapped.

usage: python src/efdecision.py eval                (cross-fitted validation vs the tuned global threshold)
       python src/efdecision.py submit <out_dir>    (fit on all validation, write a test submission)
"""
import json
import os
import sys

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from common import CACHE_DIR, f05_eval, is_val, load_sources, load_truth_pairs, write_submission
from compare_val import per_entity_f

SEED = 20260925
C_GRID = [0.0, 0.05, 0.1, 0.2, 0.3]
BASE_THR = 0.935
FLOOR = 0.30  # scores below this are treated as q = 0 (they are never near a decision boundary)


def resolve_all(pred):
    """Each record keeps only its best-scoring claimant (ties -> lowest S1 id)."""
    return (pred.sort(["s23", "p", "s1"], descending=[False, True, False])
            .unique(subset="s23", keep="first", maintain_order=True))


def fit_calibrator(pred, truth):
    y = pred.join(truth.with_columns(pl.lit(1).alias("y")), on=["s1", "s23"], how="left")["y"].fill_null(0)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    iso.fit(pred["p"].to_numpy(), y.to_numpy())
    return iso


def decide(pred, iso, C):
    """pred: resolved pairs (s1, s23, p). Returns the selected pairs."""
    d = pred.with_columns(pl.Series("q", iso.predict(pred["p"].to_numpy()).astype(np.float64)))
    d = d.sort(["s1", "q", "s23"], descending=[False, True, False]).with_columns(
        pl.col("q").cum_sum().over("s1").alias("cq"),
        pl.int_range(1, pl.len() + 1).over("s1").alias("k"),
        pl.col("q").sum().over("s1").alias("sq"),
        (1 - pl.col("q")).log().sum().over("s1").alias("log_g0"),
    ).with_columns((1.25 * pl.col("cq") / (0.25 * (pl.col("sq") + C) + pl.col("k"))).alias("gk"))
    best = d.group_by("s1").agg(
        pl.col("gk").max().alias("gmax"), pl.col("k").filter(pl.col("gk") == pl.col("gk").max()).min().alias("kbest"),
        pl.col("log_g0").first().alias("log_g0"))
    best = best.with_columns((pl.col("log_g0").exp() * np.exp(-C)).alias("g0"))
    keep = best.filter(pl.col("gmax") > pl.col("g0")).select("s1", "kbest")
    return d.join(keep, on="s1").filter(pl.col("k") <= pl.col("kbest")).select("s1", "s23", "p")


def load_val():
    truth, all_s1 = load_truth_pairs("train")
    vid = all_s1.filter(pl.Series(is_val(all_s1)))
    tv = truth.join(pl.DataFrame({"s1": vid}), on="s1", how="semi")
    va = pl.read_parquet(os.path.join(CACHE_DIR, "val_pred_s2.parquet")).filter(pl.col("p") >= FLOOR)
    return tv, vid, va


def cmd_eval():
    tv, vid, va = load_val()
    half = pl.DataFrame({"s1": vid}).with_columns((pl.col("s1").hash(SEED) % 2).alias("h"))
    va = va.join(half, on="s1")
    tot_new, tot_base, n_all = 0.0, 0.0, 0
    for h in (0, 1):
        fit, ev = va.filter(pl.col("h") != h).drop("h"), va.filter(pl.col("h") == h).drop("h")
        ids_fit = half.filter(pl.col("h") != h)["s1"]
        ids_ev = half.filter(pl.col("h") == h)["s1"]
        iso = fit_calibrator(fit, tv)
        rfit = resolve_all(fit)
        scores = {C: f05_eval(decide(rfit, iso, C), tv, ids_fit, verbose=False)["F05"] for C in C_GRID}
        C = max(scores, key=scores.get)
        new = f05_eval(decide(resolve_all(ev), iso, C), tv, ids_ev, verbose=False)
        base = f05_eval(resolve_all(ev.filter(pl.col("p") >= BASE_THR)), tv, ids_ev, verbose=False)
        print(f"half {h}: C={C} (fit-half F {scores[C]:.5f}) | held-out half: expected-F rule {new['F05']:.5f}"
              f" vs global threshold {base['F05']:.5f} ({new['F05'] - base['F05']:+.5f})  "
              f"P {new['P_macro']:.4f}/{base['P_macro']:.4f}  R {new['R_macro']:.4f}/{base['R_macro']:.4f}  "
              f"singleton acc {new['singleton_acc']:.4f}/{base['singleton_acc']:.4f}")
        tot_new += new["F05"] * len(ids_ev)
        tot_base += base["F05"] * len(ids_ev)
        n_all += len(ids_ev)
    print(f"CROSS-FITTED validation F0.5: expected-F rule {tot_new / n_all:.5f} vs global threshold "
          f"{tot_base / n_all:.5f} ({(tot_new - tot_base) / n_all:+.5f})")
    # paired bootstrap of the held-out difference is done by compare_val on the written predictions


def cmd_submit(out_dir):
    tv, vid, va = load_val()
    iso = fit_calibrator(va, tv)
    rva = resolve_all(va)
    scores = {C: f05_eval(decide(rva, iso, C), tv, vid, verbose=False)["F05"] for C in C_GRID}
    C = max(scores, key=scores.get)
    print(f"fit on all validation: C={C}, in-sample F0.5 {scores[C]:.5f}")
    s1, _ = load_sources("test")
    te = pl.read_parquet(os.path.join(CACHE_DIR, "test_pred_s2.parquet"), columns=["s1", "s23", "p_avg"])
    cand = te.select("s1", "s23")
    te = te.rename({"p_avg": "p"}).filter(pl.col("p") >= FLOOR)
    match = decide(resolve_all(te), iso, C)
    write_submission(s1["entity_id"], match, cand, out_dir)
    n = match["s1"].n_unique()
    print(f"test: {match.height:,} matches for {n:,} of {s1.height:,} S1 ({s1.height - n:,} predicted singleton)")
    with open(os.path.join(out_dir, "efdecision.json"), "w") as fh:
        json.dump({"C": C, "val_in_sample_F05": scores[C], "calibration_x": iso.X_thresholds_.tolist(),
                   "calibration_y": iso.y_thresholds_.tolist()}, fh)


if __name__ == "__main__":
    {"eval": cmd_eval, "submit": lambda: cmd_submit(sys.argv[2])}[sys.argv[1]]()
