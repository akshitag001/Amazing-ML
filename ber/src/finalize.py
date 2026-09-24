"""Stage 5: scoring, conflict resolution, threshold tuning, submission files.

1. Score candidate pairs with the trained LightGBM model.
2. Conflict resolution (each S2/S3 record belongs to at most one S1): among S1 entities whose score for a
   record clears the threshold, only the highest-scoring S1 keeps it (ties -> lowest S1 id, deterministic).
3. Threshold is tuned on the validation split for macro F0.5, on POST-resolution predictions. Validation
   conflicts are resolved against every competing S1 (train-split S1 entities included), so the competition
   is the same as on test.
4. Test: write output/matching_results.tsv and output/candidate_pairs.tsv (all pairs the model scored).

usage: python src/finalize.py
"""
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from common import CACHE_DIR, OUT_DIR, f05_eval, is_val, load_sources, load_truth_pairs, write_submission
from features import feat_dir
from train import MODEL_DIR, feature_cols

MODEL = os.path.join(MODEL_DIR, "lgb_v1.txt")
THRESHOLDS = [round(x, 3) for x in np.arange(0.30, 0.991, 0.01)]


def score(mode, filter_expr=None, tag=""):
    """Predict every candidate pair of `mode` (optionally filtered). Returns frame s1, s23, p."""
    model = lgb.Booster(model_file=MODEL)
    feats = feature_cols()
    out = []
    for f in sorted(glob.glob(os.path.join(feat_dir(mode), "part_*.parquet"))):
        lf = pl.scan_parquet(f)
        if filter_expr is not None:
            lf = lf.filter(filter_expr)
        d = lf.select(["s1", "s23"] + feats).collect()
        if d.height == 0:
            continue
        p = model.predict(d.select(feats).to_numpy().astype(np.float32), num_threads=30)
        out.append(d.select("s1", "s23").with_columns(pl.Series("p", p.astype(np.float32))))
    res = pl.concat(out)
    print(f"  scored {res.height:,} {mode} pairs {tag}", flush=True)
    return res


def resolve(pred, thr):
    """Keep pairs with p >= thr; each s23 goes only to its best-scoring S1."""
    return (pred.filter(pl.col("p") >= thr)
            .sort(["s23", "p", "s1"], descending=[False, True, False])
            .unique(subset="s23", keep="first", maintain_order=True))


def tune():
    truth, all_s1 = load_truth_pairs("train")
    val_mask = pl.Series(is_val(all_s1))
    val_ids = all_s1.filter(val_mask)
    val_pred = pl.read_parquet(os.path.join(CACHE_DIR, "val_pred_v1.parquet"))
    # competitors: train-split pairs that share a record with any validation pair
    s23_val = val_pred.select("s23").unique()
    comp = score("train", (pl.col("is_val") == 0) & pl.col("s23").is_in(s23_val["s23"].implode()),
                 tag="(train-split competitors of validation records)")
    allp = pl.concat([val_pred, comp])
    rows = []
    for thr in THRESHOLDS:
        raw = f05_eval(val_pred.filter(pl.col("p") >= thr), truth, val_ids, verbose=False)
        res = f05_eval(resolve(allp, thr), truth, val_ids, verbose=False)
        rows.append({"thr": thr, "F05_raw": raw["F05"], "F05_resolved": res["F05"], "P_resolved": res["P_macro"],
                     "R_resolved": res["R_macro"], "singleton_acc": res["singleton_acc"]})
    sweep = pl.DataFrame(rows)
    best = sweep.sort("F05_resolved", descending=True).row(0, named=True)
    best_raw = sweep.sort("F05_raw", descending=True).row(0, named=True)
    at05 = sweep.filter(pl.col("thr") == 0.5).row(0, named=True)
    with pl.Config(tbl_rows=100, float_precision=4):
        print(sweep.filter((pl.col("thr") * 100).round() % 5 == 0))
    # conflict statistics at the chosen threshold
    above = allp.filter(pl.col("p") >= best["thr"])
    contested = above.group_by("s23").len().filter(pl.col("len") > 1)
    print(f"\nat threshold 0.5: raw F0.5 {at05['F05_raw']:.4f} | resolved {at05['F05_resolved']:.4f}")
    print(f"best raw threshold {best_raw['thr']}: F0.5 {best_raw['F05_raw']:.4f}")
    print(f"BEST (resolved) threshold {best['thr']}: F0.5 {best['F05_resolved']:.4f}  P {best['P_resolved']:.4f}"
          f"  R {best['R_resolved']:.4f}  singleton acc {best['singleton_acc']:.4f}")
    print(f"conflicts at best threshold: {contested.height:,} records claimed by >1 S1 "
          f"({int(contested['len'].sum() - contested.height):,} claims removed)")
    print("full metrics at best threshold (post-resolution):")
    f05_eval(resolve(allp, best["thr"]), truth, val_ids)
    with open(os.path.join(MODEL_DIR, "threshold.json"), "w") as f:
        json.dump({"threshold": best["thr"], "val_F05": best["F05_resolved"], "sweep": rows}, f, indent=1)
    return best["thr"]


def make_submission(thr):
    t = time.time()
    s1, _ = load_sources("test")
    pred = score("test")
    match = resolve(pred, thr)
    os.makedirs(OUT_DIR, exist_ok=True)
    write_submission(s1["entity_id"], match, pred.select("s1", "s23"), OUT_DIR)
    n_matched = match["s1"].n_unique()
    print(f"test: {match.height:,} matches for {n_matched:,} of {s1.height:,} S1 entities "
          f"({s1.height - n_matched:,} predicted singleton) | written in {time.time() - t:.0f}s")


def main():
    t0 = time.time()
    thr = tune()
    print(f"tuning done in {time.time() - t0:.0f}s")
    make_submission(thr)
    print(f"FINALIZE total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
