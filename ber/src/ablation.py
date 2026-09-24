"""Leakage ablation: refit a fast LightGBM (same seeded sample, same config) with feature subsets removed
and compare validation average precision. A collapse (AP ~0.5-0.6) after removing one feature would indicate
label leakage; a modest drop means the signal is distributed.

usage: python src/ablation.py <top_feature>
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from train import PARAMS, SEED, build_train_sample, feature_cols, feature_parts

COMPETITION = ["c1_rank", "c1_n", "c1_n_exactname", "c1_margin", "c23_rank", "c23_n", "c23_n_exactname",
               "c23_margin", "raw_score", "b_cos", "b_rank_fwd", "b_rank_rev", "b_n_keys"]
QUICK = dict(PARAMS, learning_rate=0.15, num_leaves=127)


def main(top):
    feats = feature_cols()
    tr = build_train_sample(feats)
    # fixed 25% of the training sample and ~25% of validation S1 entities, both by seeded hash
    tr = tr.with_row_index("r").filter(pl.col("r").hash(SEED) % 4 == 0).drop("r")
    va = (pl.scan_parquet(feature_parts("train")).filter(pl.col("is_val") == 1)
          .filter(pl.col("s1").hash(SEED) % 4 == 0).select(feats + ["y"]).collect())
    y, w = tr["y"].to_numpy(), tr["w"].to_numpy()
    spw = w[y == 0].sum() / w[y == 1].sum()
    yv = va["y"].to_numpy()
    print(f"ablation: train {tr.height:,} rows, val {va.height:,} pairs")
    name_addr = [f for f in feats if f not in COMPETITION]
    runs = {
        "all features": feats,
        f"minus top-1 ({top})": [f for f in feats if f != top],
        "minus competition+blocking group": name_addr,
        "name features only": [f for f in feats if f.startswith("n_") or f.startswith("name_") or f == "legal_eq"],
        "address features only": [f for f in feats if f.startswith(("a_", "h_", "loc_", "nums_", "region_",
                                                                     "landmark"))],
    }
    for label, fs in runs.items():
        t = time.time()
        d = lgb.Dataset(tr.select(fs).to_numpy().astype(np.float32), label=y, weight=w, feature_name=fs)
        dv = lgb.Dataset(va.select(fs).to_numpy().astype(np.float32), label=yv, reference=d)
        m = lgb.train(dict(QUICK, scale_pos_weight=spw), d, num_boost_round=400, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(30, first_metric_only=True, verbose=False)])
        p = m.predict(va.select(fs).to_numpy().astype(np.float32), num_iteration=m.best_iteration)
        print(f"  {label:38s} features={len(fs):2d}  val AP={average_precision_score(yv, p):.5f}  "
              f"rounds={m.best_iteration}  ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
