"""Stage 4: LightGBM matcher with stratified hard-negative mining.

Training set = train-split S1 entities only (is_val == 0):
  all positives, plus negatives drawn per stratum with a seeded hash of the pair ids (row-order independent):
    H1 same core name + blank address                     all,   boost x2
    H2 high name similarity (>=90) + contradicting address  sampled, boost x1.5
    H3 top competitors (S1-side rank<=3 or record-side rank<=2), not H2   sampled, boost x1.5
    E  everything else (easy)                               sampled, boost x1
  negative weight = boost / sampling_rate (keeps the true mass of easy negatives -> calibration),
  classes balanced with scale_pos_weight = sum(w_neg) / sum(w_pos).
Early stopping on the full validation split (is_val == 1), never a random subsplit of train.

usage: python src/train.py
"""
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from common import CACHE_DIR, f05_eval, is_val, load_truth_pairs
from features import feat_dir

SEED = 20260925
MAX_ROUNDS = 700  # AP gain beyond ~700 rounds was < 1e-4 (v1 run: 0.99935 @500 -> 0.99940 @1050)
META = ["i1", "i23", "s1", "s23", "y", "is_val"]
MODEL_DIR = os.path.join(CACHE_DIR, "models")
STRATA = {  # name: (max sampled, boost)
    "H1": (None, 2.0),
    "H2": (4_000_000, 1.5),
    "H3": (3_000_000, 1.5),
    "E": (3_000_000, 1.0),
}
PARAMS = dict(
    objective="binary", learning_rate=0.1, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, num_threads=30,
    seed=SEED, bagging_seed=SEED, feature_fraction_seed=SEED, data_random_seed=SEED,
    deterministic=True, force_row_wise=True, metric=["average_precision", "binary_logloss"],
    first_metric_only=True, verbose=-1,
)


def feature_parts(mode):
    return sorted(glob.glob(os.path.join(feat_dir(mode), "part_*.parquet")))


def feature_cols(mode="train"):
    return [c for c in pl.read_parquet_schema(feature_parts(mode)[0]) if c not in META]


def stratum_expr():
    ns = pl.max_horizontal("n_tset", "n_skel_tset")
    h1 = (pl.col("n_exact_core") == 1) & (pl.col("a_blank_any") == 1)
    h2 = (ns >= 90) & (pl.col("a_blank_any") == 0) & ((pl.col("a_tset") < 70) | (pl.col("h_eq") == 0))
    h3 = (pl.col("c1_rank") <= 3) | (pl.col("c23_rank") <= 2)
    return (pl.when(pl.col("y") == 1).then(pl.lit("POS")).when(h1).then(pl.lit("H1"))
            .when(h2).then(pl.lit("H2")).when(h3).then(pl.lit("H3")).otherwise(pl.lit("E")))


def build_train_sample(feats):
    lf = pl.scan_parquet(feature_parts("train")).filter(pl.col("is_val") == 0).with_columns(
        stratum_expr().alias("stratum"),
        (pl.concat_str("s1", pl.lit("|"), "s23").hash(SEED) % 1_000_000_007 / 1_000_000_007.0).alias("u"),
    )
    counts = dict(lf.group_by("stratum").len().collect().iter_rows())
    rates = {"POS": 1.0}
    for s, (cap, _) in STRATA.items():
        rates[s] = 1.0 if cap is None else min(1.0, cap / max(counts.get(s, 1), 1))
    boost = {"POS": 1.0, **{s: b for s, (_, b) in STRATA.items()}}
    rate_expr = pl.col("stratum").replace_strict(rates, return_dtype=pl.Float64)
    df = (lf.filter(pl.col("u") < rate_expr)
          .with_columns((pl.col("stratum").replace_strict(boost, return_dtype=pl.Float64) / rate_expr)
                        .alias("w"))
          .select(feats + ["y", "w", "stratum"]).collect())
    print("strata (pool -> sampled, rate):")
    for s in ["POS"] + list(STRATA):
        print(f"  {s:>3}: {counts.get(s, 0):>11,} -> {int((df['stratum'] == s).sum()):>10,}  rate {rates[s]:.3f}"
              f"  boost {boost[s]}")
    return df


def load_split(mode, feats, val_only=False):
    lf = pl.scan_parquet(feature_parts(mode))
    if val_only:
        lf = lf.filter(pl.col("is_val") == 1)
    return lf.select(["s1", "s23"] + feats + (["y"] if mode == "train" else [])).collect()


def main():
    t0 = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)
    feats = feature_cols()
    tr = build_train_sample(feats)
    w = tr["w"].to_numpy()
    y = tr["y"].to_numpy()
    spw = w[y == 0].sum() / w[y == 1].sum()
    print(f"train sample {tr.height:,} rows | pos {int(y.sum()):,} | scale_pos_weight {spw:.2f}")
    dtr = lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), label=y, weight=w,
                      feature_name=feats, free_raw_data=True)
    del tr
    va = load_split("train", feats, val_only=True)
    dva = lgb.Dataset(va.select(feats).to_numpy().astype(np.float32), label=va["y"].to_numpy(),
                      reference=dtr, free_raw_data=False)
    print(f"validation {va.height:,} pairs | loaded in {time.time() - t0:.0f}s")
    params = dict(PARAMS, scale_pos_weight=spw)
    t = time.time()
    model = lgb.train(params, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=True),
                                 lgb.log_evaluation(50)])
    print(f"trained {model.best_iteration} rounds in {time.time() - t:.0f}s")
    model.save_model(os.path.join(MODEL_DIR, "lgb_v1.txt"), num_iteration=model.best_iteration)

    # ---- checkpoint signal: default threshold 0.5 on the validation split
    p = model.predict(dva.get_data(), num_iteration=model.best_iteration, num_threads=30)
    pred = va.select("s1", "s23").with_columns(pl.Series("p", p.astype(np.float32)))
    pred.write_parquet(os.path.join(CACHE_DIR, "val_pred_v1.parquet"))
    truth, all_s1 = load_truth_pairs("train")
    val_ids = all_s1.filter(pl.Series(is_val(all_s1)))
    print("VALIDATION @ threshold 0.5:")
    res = f05_eval(pred.filter(pl.col("p") >= 0.5), truth, val_ids)
    gain = model.feature_importance("gain")
    imp = sorted(zip(feats, gain), key=lambda x: -x[1])
    with open(os.path.join(MODEL_DIR, "lgb_v1_meta.json"), "w") as f:
        json.dump({"params": params, "best_iteration": model.best_iteration, "val_at_0.5": res,
                   "importance_gain": imp}, f, indent=1, default=float)
    tot = gain.sum()
    print("feature importance (gain share):")
    for name, g in imp:
        print(f"  {name:22s} {100 * g / tot:6.2f}%")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
