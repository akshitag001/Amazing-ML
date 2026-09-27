"""Re-run the TEST side only with the already-trained models (no retraining).

Used when a change affects only how test records are read (e.g. region imputation): train features, fold
models, stage-2 models, variant and threshold stay exactly as validated. Steps: stage-1 p1 from the saved
fold models -> context features -> stage-2 scores -> conflict resolution at the saved threshold -> submission.

usage: python src/retest.py <out_dir>          (run after: prepare.py test, blocking.py test, features.py test)
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

from common import CACHE_DIR, load_sources, write_submission
from finalize import resolve
from stack import N_FOLDS, build_ctx, p1_path, predict_xgb, score_both
from train import MODEL_DIR, feature_cols, feature_parts


def main(out_dir):
    t0 = time.time()
    feats = feature_cols("test")
    assert feats == feature_cols("train"), "train/test feature schema mismatch"
    models = []
    for k in range(N_FOLDS):
        m = xgb.Booster()
        m.load_model(os.path.join(MODEL_DIR, f"xgb_s1_fold{k}.json"))
        m.set_param({"device": "cuda"})
        models.append(m)
    out = []
    for f in feature_parts("test"):
        d = pl.read_parquet(f, columns=["i1", "i23"] + feats)
        X = d.select(feats).to_numpy().astype(np.float32)
        p1 = np.column_stack([predict_xgb(m, X) for m in models]).mean(axis=1)
        out.append(d.select("i1", "i23").with_columns(pl.Series("p1", p1.astype(np.float32))))
    pl.concat(out).write_parquet(p1_path("test"))
    print(f"p1 test {time.time() - t0:.0f}s", flush=True)
    build_ctx("test")
    cfg = json.load(open(os.path.join(MODEL_DIR, "threshold_s2.json")))
    var, thr = cfg["variant"], cfg["threshold"]
    te = score_both("test")
    te.write_parquet(os.path.join(out_dir, "test_pred.parquet"))
    s1, _ = load_sources("test")
    match = resolve(te.select("s1", "s23", pl.col(var).alias("p")), thr)
    write_submission(s1["entity_id"], match, te.select("s1", "s23"), out_dir)
    n = match["s1"].n_unique()
    print(f"test ({var} @ {thr}): {match.height:,} matches for {n:,} of {s1.height:,} S1 "
          f"({s1.height - n:,} predicted singleton) | total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    os.makedirs(sys.argv[1], exist_ok=True)
    main(sys.argv[1])
